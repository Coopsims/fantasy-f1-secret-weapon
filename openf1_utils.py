import time
import requests
import os
import threading
import random

def load_env_file(filepath=".env"):
    """
    Manually load environment variables from a .env file if it exists.
    Useful when python-dotenv is not installed.
    """
    if os.path.exists(filepath):
        # print(f"  📂 Loading secrets from {filepath}...")
        try:
            with open(filepath, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip().strip('"').strip("'")
        except Exception as e:
            print(f"  ⚠  Failed to load {filepath}: {e}")

# Load environment variables from .env if present at the start
load_env_file()

class OpenF1Error(Exception):
    """A request that failed (rate limit, server error, timeout...) - as opposed to a successful empty answer."""


class OpenF1Client:
    """
    Client for OpenF1 API with support for paid tokens and 1-hour refresh.

    get()         legacy: returns [] when the request fails (indistinguishable from "no data")
    get_strict()  raises OpenF1Error on failure; [] means OpenF1 really had no rows. Use this for anything cached.
    Pacing: set `min_interval` (seconds between requests, process-wide). It doubles after each 429 (up to 3 s)
    and relaxes back towards `base_interval` after a run of successes. See docs/OPENF1_NOTES.md.
    """
    def __init__(self, token=None, refresh_fn=None, base_url="https://api.openf1.org/v1"):
        self.token = token
        self.refresh_fn = refresh_fn
        self.last_refresh = 0  # Initialize to 0 to ensure immediate check on first request
        self.expiry_duration = 3600  # 1 hour
        self.base_url = base_url
        self._lock = threading.Lock()
        self.base_interval = 0.0
        self.min_interval = 0.0
        self._next_ok = 0.0
        self._ok_streak = 0

    def set_token(self, token):
        with self._lock:
            self.token = token
            self.last_refresh = time.time()

    def check_refresh(self, force=False):
        """Check if the token needs refreshing (every hour or if missing)."""
        if not self.refresh_fn:
            return

        now = time.time()
        # Refresh if:
        # 1. Forced
        # 2. Token is missing and we haven't tried in the last 30 seconds
        # 3. Token is expired (1 hour)
        needs_refresh = force or \
                        (self.token is None and now - self.last_refresh >= 30) or \
                        (now - self.last_refresh >= self.expiry_duration)

        if needs_refresh:
            with self._lock:
                # Double-check
                needs_refresh = force or \
                                (self.token is None and now - self.last_refresh >= 30) or \
                                (now - self.last_refresh >= self.expiry_duration)
                if needs_refresh:
                    print("\n  🔄 Fetching/Refreshing OpenF1 paid token...")
                    try:
                        new_token = self.refresh_fn()
                        if new_token:
                            self.token = new_token
                            self.last_refresh = time.time()
                            print("  ✅ OpenF1 token obtained successfully.")
                        else:
                            # If refresh_fn returned None, credentials might be missing or invalid.
                            # Set last_refresh to current time to avoid spamming.
                            self.last_refresh = time.time()
                    except Exception as e:
                        print(f"  ⚠  Failed to refresh OpenF1 token: {e}")
                        self.last_refresh = time.time()

    # ---- pacing -------------------------------------------------------
    def set_pacing(self, seconds):
        self.base_interval = self.min_interval = float(seconds)

    def _pace(self):
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            at = max(now, self._next_ok)
            self._next_ok = at + self.min_interval
        if at > now:
            time.sleep(at - now)

    def _slow_down(self):
        with self._lock:
            self.min_interval = min(max(self.min_interval * 2, 0.5), 3.0)
            self._ok_streak = 0

    def _success(self):
        with self._lock:
            self._ok_streak += 1
            if self._ok_streak >= 25 and self.min_interval > self.base_interval:
                self.min_interval = max(self.base_interval, self.min_interval * 0.8)
                self._ok_streak = 0

    def get_strict(self, endpoint, params=None, retries=6, timeout=90):
        """JSON rows for the query. [] = OpenF1 answered "no rows" (HTTP 404). Raises OpenF1Error otherwise."""
        url = f"{self.base_url}/{endpoint}"
        last = "unknown"
        for attempt in range(retries):
            self.check_refresh()
            self._pace()
            headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
            try:
                r = requests.get(url, params=params, headers=headers, timeout=timeout)
            except requests.RequestException as e:
                last = type(e).__name__
                time.sleep(min(2 ** attempt, 30))
                continue
            if r.status_code == 404:
                self._success()
                return []
            if r.status_code == 429:
                try:
                    wait = float(r.headers.get("Retry-After"))
                except (TypeError, ValueError):
                    wait = min(2 ** (attempt + 1), 60)
                wait += random.uniform(0, 1)
                self._slow_down()
                last = "HTTP 429"
                print(f"  ⚠  OpenF1 rate limited ({endpoint}, attempt {attempt+1}/{retries}); pacing now "
                      f"{self.min_interval:.1f}s, sleeping {wait:.0f}s")
                time.sleep(wait)
                continue
            if r.status_code == 401 and self.refresh_fn:
                self.check_refresh(force=True)
                last = "HTTP 401"
                continue
            if r.status_code >= 500:
                last = f"HTTP {r.status_code}"
                time.sleep(min(2 ** attempt, 30))
                continue
            if r.status_code >= 400:
                raise OpenF1Error(f"{endpoint} {params}: HTTP {r.status_code} {r.text[:120]}")
            try:
                rows = r.json()
            except ValueError:
                last = "invalid JSON"
                continue
            self._success()
            return rows
        raise OpenF1Error(f"{endpoint} {params}: gave up after {retries} attempts ({last})")

    def get(self, endpoint, params=None, retries=3):
        self.check_refresh()
        url = f"{self.base_url}/{endpoint}"

        headers = {}
        # Grab current token to ensure consistency during this request
        current_token = self.token
        if current_token:
            # Paid version usually expects the token in the headers
            headers["Authorization"] = f"Bearer {current_token}"

        for attempt in range(retries):
            try:
                # Use a session if multiple requests are made frequently
                r = requests.get(url, params=params, headers=headers, timeout=15)

                if r.status_code == 404:
                    return []

                if r.status_code == 429:
                    # If we have a paid token, 429 might still happen but should be rare
                    wait = 2 * (attempt + 1)
                    print(f"  ⚠  OpenF1 rate limited (attempt {attempt+1}/{retries}). Sleeping {wait}s...")
                    time.sleep(wait)
                    continue

                if r.status_code == 401:
                    print("  ⚠  OpenF1 token invalid or expired. Attempting refresh...")
                    if self.refresh_fn:
                        self.token = self.refresh_fn()
                        self.last_refresh = time.time()
                        # Retry with new token
                        headers["Authorization"] = f"Bearer {self.token}"
                        continue

                r.raise_for_status()
                return r.json()
            except Exception as e:
                if attempt == retries - 1:
                    print(f"  ⚠  OpenF1 request failed after {retries} attempts ({endpoint}): {e}")
                    return []
                time.sleep(1)
        return []

# Example token refresh function (placeholder for the user to implement their specific logic)
def fetch_token(username, password):
    """
    Obtain a new access token from OpenF1 via POST request.
    """
    if not username or not password:
        return None

    token_url = "https://api.openf1.org/token"
    payload = {
        "username": username,
        "password": password
    }
    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }

    try:
        # User provided logic
        response = requests.post(token_url, data=payload, headers=headers, timeout=10)

        if response.status_code == 200:
            token_data = response.json()
            return token_data.get('access_token')
        else:
            print(f"  ⚠  Error obtaining token: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"  ⚠  Exception during token fetch: {e}")
        return None

def default_token_creator():
    """
    Fetches a new token using username and password from environment variables.
    """
    username = os.environ.get("OPENF1_USERNAME")
    password = os.environ.get("OPENF1_PASSWORD")

    if username and password:
        return fetch_token(username, password)

    # Fallback to direct token if available
    return os.environ.get("OPENF1_PAID_TOKEN")

# Global instance for easy access across the project
openf1_client = OpenF1Client(refresh_fn=default_token_creator)
