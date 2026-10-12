# OpenF1: rate limits and how to pull data (notes from the 2026-10 work)

What was **measured** is marked as such; what is a **guess** says so. The docs at https://openf1.org/docs/ do not state any
rate limit, response-size limit or pagination, and the API sends no rate-limit headers (only `Content-Encoding: gzip`).

## Rate limits

* **Stated by the user (2026-10-12): 6 requests per second and 60 per minute.** The per-minute cap is the binding one
  for long pulls - about 1 request per second sustained - which matches everything observed below.

* Limits are undocumented; I could not read them from the API (no rate-limit headers on normal responses).
  **Observed:**
  * 8 parallel worker threads (the default of `f1_trainer.collect_data` with a token) produced frequent HTTP 429s.
  * Strictly sequential at 0.35 s spacing: the first 429 came after about 6 weekends (roughly 100 requests in about
    3 minutes). The client slept about 60 s, i.e. the 429 carried a `Retry-After` of about 60 s (inferred from the sleep;
    the raw header was not logged).
  * Sequential at **1.1 s spacing: 81 weekends (15 requests each, about 1,200 in total; 39 MB gzipped for all 87 weekends) in 25 minutes with zero 429s.**
  * Where the ceiling lies between 0.35 s and 1.1 s was **not tested**; 1.1 s is a known-good setting, not a measured optimum.
* So: **pull sequentially, pace at about 1 s, never fan out across threads.** `openf1_client.set_pacing(1.1)` sets a
  process-wide minimum gap; after a 429 the client doubles it (max 3 s), honours `Retry-After`, adds jitter, and relaxes
  back after 25 successes. `fetch_openf1_full.py --pace` sets it (default 1.1 s).
* A 404 means "no rows match" and is a legitimate empty answer. Everything else that fails must **not** be treated as empty.

## The trap that cost the most time

`openf1_client.get()` returns `[]` both for "no data" and for "request failed after retries". Callers that cached the
`[]` froze failed pulls as final (2024 Qatar and Abu Dhabi, 2025 Monza and 2026 Zandvoort came back empty and stayed
empty). Use `openf1_client.get_strict()` (raises `OpenF1Error`) for anything that is cached, and record a status per
request (`fetch_openf1_full.py` stores `ok` / `empty` / `error: ...` per endpoint and retries only the errors).

## Get everything in one pull

* **Measured:** a request with `meeting_key=` returns exactly the rows of all its sessions combined (identical counts for
  drivers, laps, position, stints, pit, weather, race_control, session_result, starting_grid, overtakes on 2026 Australia).
  So use one request per endpoint per weekend, not one per session: about 14 requests per weekend instead of 30+.
* `python fetch_openf1_full.py` does that for 2023-2026 into `openf1_full/<year>_<meeting_key>.json.gz` (gitignored),
  including `intervals` (87 weekends, 39 MB gzipped on disk). It is resumable, retries only failed endpoints and weekends still
  in progress, and took about 25 minutes for 81 weekends at 1.1 s spacing. `openf1_cache.py` rebuilds the existing raw formats from it.
* **Size per race weekend (measured, 2026 Australia, uncompressed JSON):** laps 1.3 MB, position 0.4 MB, intervals 3.2 MB,
  everything else under 0.15 MB each. `--no-intervals` skips the biggest.
* Not pulled: `car_data` and `location` (per-sample telemetry; far larger, size not measured). Pull them per session
  and per driver with date filters if ever needed.

## Data gotchas found on the way

* `drivers` queried with the **Practice 1** `session_key` can list reserve drivers instead of the race drivers: take the
  line-up from each session's own `drivers` (see `f1_trainer.get_session_data`).
* Sprint sessions have `session_type` "Race" and "Qualifying"; tell them apart by `session_name`.
* `pit.stop_duration` is null for many 2026 races (only `lane_duration`), so constructor pit points are unknown there.
* 2023: the only weekend with no data is Imola (cancelled). Spielberg, Singapore and São Paulo 2023 looked like "source gaps"
  in the old per-session pulls but were failed requests that were cached as final; the one-pull cache has them.
* **Sprint weekends have two sessions typed "Qualifying" and two typed "Race".** Use `session_name` ("Sprint Qualifying",
  "Sprint Shootout", "Sprint", "Qualifying", "Race"). `session_key` order is NOT chronological: the real Qualifying has the
  lower key in 2023 and 2024, the Sprint Qualifying has the lower key in 2025 and 2026. The old key-order guess swapped
  Qualifying and Sprint Qualifying for the 12 Sprint weekends of 2023-24 (see results/DATA_CORRECTION.md).
* The `position` stream's last record matches the official `session_result` in 99-100% of qualifying positions and 93-99% of race
  positions (differences are mostly post-race penalties and retirements); pole/winner match in all 222 sessions compared.
* Tokens last about an hour (`check_refresh` handles it). Credentials live in `.env` (`OPENF1_USERNAME`, `OPENF1_PASSWORD`); never print or commit them.

## Team lock (needed to decide which sessions are usable features)

Start of Qualifying; on Sprint weekends the start of the Sprint (formula1.com 2026 strategist previews, matching OpenF1
`date_start` exactly for Australia, China, Japan, Miami and Canada). See `fetch_practice.py` / `openf1_cache.lock_time`.
