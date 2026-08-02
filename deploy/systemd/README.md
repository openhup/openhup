# Bare-metal install with systemd

Containers are the easy path; this is the other one, and it is a first-class option rather than an
afterthought. It uses less memory, starts faster, and gives you `journalctl` and systemd's sandboxing
instead of Docker's.

## Layout these units assume

```
/opt/openhup/backend/         the repo's backend/,        with .venv inside
/opt/openhup/vision-service/  the repo's vision-service/, with .venv inside
/etc/openhup/                 config.yaml, cameras.yaml, vision.yaml, openhup.env (0600)
/var/lib/openhup/             state, snapshots, models
```

## Install

```sh
sudo useradd --system --home /var/lib/openhup --shell /usr/sbin/nologin openhup
sudo mkdir -p /opt/openhup /etc/openhup /var/lib/openhup/{snapshots,models}
sudo chown -R openhup:openhup /var/lib/openhup

sudo git clone https://github.com/openhup/openhup /opt/openhup
cd /opt/openhup

# Each service gets its own venv; they have deliberately different dependency sets.
sudo -u openhup sh -c 'cd backend && uv sync --frozen'
sudo -u openhup sh -c 'cd vision-service && uv sync --frozen --extra openvino'  # or cpu / cuda

sudo cp config/config.yaml.example /etc/openhup/config.yaml
sudo cp config/vision.yaml.example /etc/openhup/vision.yaml
sudo cp examples/cameras/cameras.yaml examples/personalities/personalities.yaml /etc/openhup/
sudo cp deploy/env/openhup.env.example /etc/openhup/openhup.env
sudo chmod 600 /etc/openhup/openhup.env
sudo chown openhup:openhup /etc/openhup/openhup.env
sudoedit /etc/openhup/openhup.env   # set the database password and camera credentials

sudo -u openhup /opt/openhup/vision-service/.venv/bin/python \
    -m openhup_vision.backends --fetch

sudo cp deploy/systemd/openhup-*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now openhup-api openhup-engine openhup-vision
```

## Verify

```sh
systemctl status openhup-api openhup-engine openhup-vision
journalctl -u openhup-engine -f
curl -s localhost:8080/api/v1/system/health | jq

# Check the sandbox is actually applied. Under 3.0 is good; the API unit scores around 1.5.
systemd-analyze security openhup-api.service
```

## The two settings people get wrong

**`IPAddressAllow`.** All three units deny egress by default and allow only loopback and RFC1918.
That is what makes "nothing leaves your network" enforced by the init system rather than merely
intended. If you use a remote LLM, or a notification channel on the internet (ntfy.sh, Discord, a
hosted Matrix homeserver), the relevant unit needs `IPAddressAllow=any` - and you should know that is
what you are doing. Self-hosting ntfy avoids the question entirely.

**`SupplementaryGroups=render video`** on the vision unit. Without it there is no `/dev/dri` access,
OpenVINO silently falls back to CPU, and you spend an afternoon wondering why your iGPU is idle.
Confirm the group name on your distro with `stat -c '%G' /dev/dri/renderD128` - it is `render` on
Debian and Ubuntu, `video` on some others.

## Updating

```sh
cd /opt/openhup && sudo git pull
sudo -u openhup sh -c 'cd backend && uv sync --frozen'
sudo systemctl restart openhup-api openhup-engine openhup-vision
```

Migrations run automatically from the API unit's `ExecStartPre`. Restart the API before the engine if
you are doing it by hand: the engine expects the schema to already be current.
