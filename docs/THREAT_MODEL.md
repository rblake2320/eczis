# UNFILED-IP
# Threat Model

Covered in v0.1.1 (each line maps to tests in `RELEASE_CLAIMS.yaml`):

- artifact mutation (single bit), deletion, insertion, manifest reorder/edit, wrong key (C3);
- attacker holding the key who re-signs a non-canonical (unsorted/duplicate) manifest (C3);
- archive path traversal, absolute or drive paths, backslashes, duplicate members, oversize members (C3);
- unknown future schema majors in manifests and captures (C12);
- interrupted sealing and interrupted final rename: no partial bundle can verify (C10);
- ledger mutation, deletion, reorder, malformed or injected records, torn tail after a crash (C11);
- ledger **tail truncation and full history rewrite**, detected only with the HMAC-sealed head anchor (C11);
- concurrent ledger writers across processes (C11);
- model self-validation through inverse crime (C1);
- detectable false-feature injection, and **K-1 single-frame contradiction** (C2, C6);
- **low-residual, non-identifiable detail** labeled R (C7);
- provenance promotion during rendering, both backends (C4);
- threshold tuning on locked-test seeds (C13);
- key overwrite and short key material (C14); keys at rest wrapped with Windows DPAPI
  (CurrentUser) when created with `eczis key create` on Windows;
- personal media, evidence bundles, or keys entering git history (`tools/evidence_guard.py`,
  pre-commit/pre-push hooks, CI).

Not covered: key theft from a live session, malicious Python runtime, compromised OS,
supply-chain attack, trusted timestamping, signer identity / third-party nonrepudiation
(HMAC proves integrity only to a key holder), or legal chain-of-custody procedure.
