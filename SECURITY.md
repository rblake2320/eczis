# UNFILED-IP
# Security Boundary

- Keep this repository private until filing authorization is recorded in `IP_STATUS.yaml`.
- `DEMO_ONLY.key` is deterministic and intentionally insecure. It exists only so the synthetic bundle can be reproduced and verified.
- Production evidence must use a random key protected by Windows DPAPI/CNG or a hardware-backed keystore.
- HMAC authenticates to parties holding the shared secret. It does not establish third-party nonrepudiation.
- The viewer is read-only and cannot reseal evidence.
