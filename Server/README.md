# Inference Server

This folder contains two server modes.

## Required VM Mode: CodeIgniter + PHP

For the deployment required by `Project.pdf`, use the VM scripts:

```powershell
VM\README.md
```

In Ubuntu VM:

```bash
./VM/setup_ubuntu.sh
./VM/setup_apache.sh
./VM/setup_socket_service.sh
```

The VM mode uses:

- Apache2 + CodeIgniter 4 / PHP for the web/backend server on port `17888`
- CodeIgniter MVC Model for socket inference calls and SQL history storage
- Python TCP socket inference server as a systemd service on port `16888`
- SQLite history database at `Server/CodeIgniterApp/writable/prediction_history.sqlite`
- URL: `http://localhost:17888/2026Project/`

## Local Flask Mode

The Flask mode is only a local development shortcut. It is useful on this Windows machine because PHP/Composer are not installed locally.

It follows the same project deployment shape from `Project.pdf`:

- HTTP site: `http://localhost:17888/2026Project/`
- Prediction socket: `127.0.0.1:16888`
- Page structure: header, main body, footer
- Model: MLP feature model + ConvTransformer ensemble

## Run Local Flask Mode

```powershell
python .\Server\run_server.py
```

Then open:

```text
http://localhost:17888/2026Project/
```

## API

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:17888/2026Project/api/predict `
  -ContentType 'application/json' `
  -Body '{"transcript_id":"demo","CDSseq":"AUGGCCAAGUAA","5UTRseq":"","3UTRseq":"","threshold":0.5}'
```

Labels:

- `0`: highly degraded mRNA
- `1`: lowly degraded / stable mRNA

## MVC History Storage

In VM mode, `app/Models/PredictionModel.php` owns the server-side inference workflow:

- sends JSON prediction requests to the Python socket server
- validates the socket response
- saves recent prediction summaries into SQLite
- provides recent history rows for the home page

## CSS

The web UI uses Tailwind CSS v4.3 through the local CLI build.

```powershell
npm.cmd install
npm.cmd run build:css
```
