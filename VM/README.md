# Ubuntu VM Deployment

This is the formal server workflow for `Project.pdf`.

- VM OS: Ubuntu 22.04
- Web server: Apache2
- Backend framework: CodeIgniter 4 / PHP
- Apache HTTP port: `17888`
- URL: `http://localhost:17888/2026Project/`
- Deep learning socket service: Python TCP server on `127.0.0.1:16888`

## Architecture

```text
Browser
  -> Apache2 :17888
  -> CodeIgniter 4 Controller /2026Project/
  -> PHP fsockopen("127.0.0.1", 16888)
  -> Python PyTorch inference socket server
  -> JSON prediction
  -> CodeIgniter renders result page
```

Apache serves only the CodeIgniter `public/` directory:

```text
Server/CodeIgniterApp/public
```

This follows CodeIgniter's deployment recommendation that the web server document root should point to `public/`, not the project root.

## 1. Copy Project Into Ubuntu VM

Put this project folder in the VM, for example:

```bash
~/Final_project
```

Then enter the folder:

```bash
cd ~/Final_project
```

## 2. Install Dependencies And Generate CodeIgniter App

```bash
chmod +x VM/*.sh
./VM/setup_ubuntu.sh
```

This installs:

- Apache2
- PHP and required PHP extensions
- Composer
- Python virtual environment
- Python ML dependencies
- CodeIgniter 4 appstarter under `Server/CodeIgniterApp`

It also copies the project Controller, Views, Routes, and Tailwind CSS into the CodeIgniter app.

## 3. Configure Apache On Port 17888

```bash
./VM/setup_apache.sh
```

This creates:

```text
/etc/apache2/sites-available/2026project.conf
```

It enables:

- Apache `rewrite` module
- Apache `headers` module
- `Listen 17888`
- VirtualHost `*:17888`
- DocumentRoot `Server/CodeIgniterApp/public`

Restart later with:

```bash
./VM/run_codeigniter.sh
```

## 4. Install Deep Learning Socket As A systemd Service

```bash
./VM/setup_socket_service.sh
```

This creates and starts:

```text
2026project-socket.service
```

Check status:

```bash
systemctl status 2026project-socket.service
```

Restart:

```bash
sudo systemctl restart 2026project-socket.service
```

If you prefer manual demo mode instead of systemd:

```bash
./VM/run_socket.sh
```

## 5. Open The Site

Inside the VM:

```text
http://localhost:17888/2026Project/
```

If VirtualBox port forwarding maps host `17888` to guest `17888`, open the same URL from the host browser:

```text
http://localhost:17888/2026Project/
```

## API Smoke Test

```bash
curl http://localhost:17888/2026Project/api/health
curl -X POST http://localhost:17888/2026Project/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"transcript_id":"demo","CDSseq":"AUGGCCAAGUAA","5UTRseq":"AUG","3UTRseq":"UUUU","threshold":0.5}'
```

## Troubleshooting

Apache logs:

```bash
sudo tail -f /var/log/apache2/2026project_error.log
```

Socket service logs:

```bash
journalctl -u 2026project-socket.service -f
```

If Apache returns `403 Forbidden`, confirm the project folder can be traversed by `www-data`. The setup script uses ACLs, but if the project was moved after setup, rerun:

```bash
./VM/setup_apache.sh
```
