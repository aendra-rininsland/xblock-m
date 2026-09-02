# infra/ — moving xblock off AWS

Runbook for replacing two idle EC2 instances (~$165/month) with two ARM VPS
(~$18/month), and adding the backups and alerting the stack currently lacks.

Written 2 September 2026. Nothing here has been applied to production yet.

## Why

Measured, not assumed:

| | queue.xblock | xblock (Ozone) |
|---|---|---|
| Instance | `r8g.large` | `t2.large` (**not** the `m3.medium` the handover claims) |
| vCPU / RAM | 2 / 16 GB | 2 / 8 GB |
| In use | 598 MB | 3.0 GB |
| Load | 0.00 | ~0.1 |
| Egress | ~51 GB/mo | ~9 GB/mo |
| ~Cost | $86/mo | $68/mo |

The handover treats leaving AWS as blocked because "the firehose consumer is
the bandwidth hog that put it there". It is not. The firehose is *inbound*, and
inbound to AWS is free. Combined egress is ~60 GB/month against a 100 GB free
allowance — about $5 at list rate if it were billed at all. The bill is
instance cost, and the instances are idle.

Both `ghcr.io/bluesky-social/ozone` and `postgres:14.11` publish **arm64**, so
the cheapest capacity is available without a rebuild.

## Files

| File | Purpose |
|---|---|
| `cloud-init-queue.yaml` | Base provision for the queue box: user, firewall, Redis bound off the internet |
| `setup-queue.sh` | Node, PM2, WireGuard server, Redis auth, app deploy. Idempotent |
| `supervisord-tunnel.conf` | Love → queue Redis tunnel, to append to `supervisord.conf` |
| `ozone-compose.yaml` | Ozone stack with **pinned digests** and Watchtower removed |
| `pg-backup.sh` | Nightly dump, **verified before upload**, offsite, with retention |
| `label-watchdog.py` | Dead man's switch on the public label stream |

## Order

Queue box first. Its state is disposable — Redis holds 8 MB, and every job
expires after 48 hours by design — so it is the cheap place to get the pattern
wrong. Ozone second: it is stateful, public, and carries the labeller's
identity.

### Hetzner gotchas

Learned by hitting all of them on 2 September 2026. The API's errors point at
the wrong field in every case.

**CAX (Ampere) is EU-only** — `nbg1` and `hel1`. Not `fsn1` (supported but shown
unavailable), and not offered in Ashburn or Hillsboro at all. An ARM plan
therefore cannot be placed in the US; a US deployment means the x86 CX/CPX line.

**`ubuntu-24.04` is two different images.** One x86 (`161547269`), one arm
(`161547270`), same name. Resolving by name yields the x86 image, and the
resulting architecture mismatch is reported as *"unsupported location for server
type"* — which sends you looking at the location for a long time. Pass the arm
image ID.

**`datacenter` is gone.** Deprecated 2025-12-16; only `location` is accepted
now. Passing a datacenter returns a clear error, unlike the two above.

**Prices are higher than commonly quoted:** CAX11 €5.99/mo and CAX21 €10.49/mo
(net = gross), plus €0.50 per primary IPv4. Both include 20 TB of traffic.

**A new account cannot create servers until it is verified.** Every
`(type, location)` pair returns *"unsupported location for server type"*, x86
included, while free resources like placement groups create fine. That
combination means the token and write access are good and compute is gated —
not that the request is wrong. Clearing it needs the account owner and Hetzner
support, not a code change.

### Phase 1 — queue box

1. Create CAX11, Ubuntu 24.04, **`nbg1`** (not `fsn1` — see above), with
   `cloud-init-queue.yaml` as user data.
2. `scp setup-queue.sh` across and run it with sudo.
3. Copy the generated Redis password out of `/root/redis-password.txt`, then delete it.
4. Add Love as a peer, or use the SSH tunnel (below).
5. Point Love's worker at the tunnel, stop the old firehose consumer.
6. Discard the in-flight queue. It is 8 MB of jobs that expire anyway.

### Phase 2 — Ozone

1. Create CAX21, Ubuntu 24.04, same location.
2. Recreate `/ozone/` — copy `caddy/`, and `{ozone,ozone_news,postgres}.env`
   **verbatim**. Do not reconstruct them by hand; they hold the signing key.
3. Deploy `ozone-compose.yaml`, restore the dump, verify (see below).
4. Drop the DNS TTL **48 hours ahead** — it is currently 14400 (4 hours) on
   Google nameservers, which would otherwise make the cutover window four hours
   of split traffic.
5. Cutover: stop Ozone, final dump, restore, flip the A record, watch labels resume.
6. Leave the old instance **stopped, not terminated**, for a week.

## Three things that must not change

**The hostname.** `xblock.aendra.dev` is in the labeller's DID document. Move
the DNS record; never the name.

**The signing key.** It lives in the `signing_key` table and in `ozone.env`.
Lose it and the labeller's identity is gone — no image backup substitutes.

**Label sequence numbers.** Consumers of `subscribeLabels` hold cursors against
`label.id`. A custom-format `pg_restore` preserves ids and sequence state; a
hand-rebuilt table would not.

## Networking

Redis is currently a public port guarded by a security-group IP allowlist,
which silently stops working whenever the home IP changes. On the new box it
binds to loopback and the WireGuard interface only.

Love reaches it through an **SSH tunnel supervised by supervisord**, not
WireGuard. WireGuard would need root on Love to install `wireguard-tools` and
bring up the interface, and sudo there is password-gated. The tunnel needs no
privileges, reuses the existing SSH key, and supervisord already provides the
restart loop. The WireGuard server side is configured by `setup-queue.sh`
anyway — the WSL2 kernel does have the module — so switching later is a
Love-side change only.

## Monitoring

Set thresholds from the **trough**, not the average. Output is strongly
diurnal: ~1,150 labels/hour at the US evening peak against 125/hour in the
small hours, with label yield swinging 73 to 16 per thousand images across the
same day. That is content mix, not the model degrading.

The window must be long for the same reason. A 90-second sample of a healthy
286/hour stream legitimately returned a single label while testing this — which
reads exactly like an outage. Five minutes is the floor; ten is better.

```
*/15 * * * * /opt/xblock/label-watchdog.py --window 600 --min-labels 5 \
    --min-classes 1 --ping-url https://hc-ping.com/<uuid>
```

Run it somewhere that is not the infrastructure it watches.

## Known issues found while surveying

**The firehose consumer crashloops.** `xblock-firehose` restarted 31 times in
41 hours. `@skyware/jetstream` emits an `error` event on websocket failure that
nothing handles, so Node turns it into an uncaught exception and PM2 restarts
the process. `ReconnectingWebSocket` underneath would have reconnected on its
own. Each restart is a gap in firehose consumption. Worth fixing *before* the
migration so the crashloop is not carried onto new hardware — and so a restart
storm after cutover is unambiguous evidence of a migration problem rather than
this.

**Watchtower auto-updates the moderation service.** See `ozone-compose.yaml`.

**No backups existed before 2 September 2026.** `pg-backup.sh` addresses it.
