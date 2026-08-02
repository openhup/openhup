# Security policy

OpenHup watches the inside of homes. We take that responsibility seriously, and we want
vulnerability reports handled privately so a fix can ship before a problem is described publicly.

## Reporting a vulnerability

Please **do not open a public issue** for a security problem.

- Use GitHub's private vulnerability reporting
  (**Security → Advisories → Report a vulnerability**) on this repository. That keeps the report
  private and lets us coordinate a fix and an advisory in one place.

Include, if you can:

- what you did, what happened, and what you expected
- the affected version (or commit)
- `GET /api/v1/system/info` and `GET /api/v1/system/health` output if relevant
- a minimal reproduction, or as much of one as you can share safely

## What to expect

This is a hobbyist-scale project with no dedicated security team and no bug bounty. What we can
promise is that reports are taken seriously, fixed in the open, and credited if you want them to be.

## Supported versions

Only the latest release on the `main` branch is supported. We do not backport fixes to older
releases.

## Out of scope

The threat model in [docs/SECURITY_PRIVACY.md](docs/SECURITY_PRIVACY.md) is the authoritative list
of what is and is not considered a vulnerability. In particular, we do not treat these as
vulnerabilities:

- Anything that requires physical access to the machine or an already-compromised network.
- Misconfiguration, including exposing the service to the internet without a reverse proxy and
  authentication (the configuration layer refuses the most dangerous cases at startup, but it
  cannot stop a deliberate `bind_host: 0.0.0.0` behind a broken proxy).
- Attacks on the cameras themselves, which are third-party devices and should be isolated on a
  VLAN as described in the installation guide.
- Essentially, if it isn't advised and you do it, then that's on you.
