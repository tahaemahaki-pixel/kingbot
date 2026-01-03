# King Bot VPS Quick-Start Guide

## How to Connect to Your VPS

1. Open Terminal (Mac/Linux) or PowerShell (Windows)
2. Type this and press Enter:
   ```
   ssh root@209.38.84.47
   ```
3. You're now connected when you see: `root@ubuntu-s-1vcpu-1gb-syd1-01:~#`

---

## Bot Commands (run these on the VPS)

| What you want to do | Command |
|---------------------|---------|
| **See live logs** | `journalctl -u kingbot -f` |
| **Stop watching logs** | Press `Ctrl+C` |
| **Check if bot is running** | `systemctl status kingbot` |
| **Stop the bot** | `systemctl stop kingbot` |
| **Start the bot** | `systemctl start kingbot` |
| **Restart the bot** | `systemctl restart kingbot` |
| **Disconnect from VPS** | Type `exit` and press Enter |

---

## Common Tasks

### Check if the bot is running
```bash
systemctl status kingbot
```
- **Active (running)** = Bot is working ✅
- **Inactive (dead)** = Bot is stopped ❌

### View what the bot is doing (live)
```bash
journalctl -u kingbot -f
```
This shows real-time updates. Press `Ctrl+C` to stop watching (bot keeps running).

### View last 50 lines of logs
```bash
journalctl -u kingbot -n 50
```

### Restart after making changes
```bash
systemctl restart kingbot
```

---

## Important Info

| Item | Value |
|------|-------|
| VPS IP | `209.38.84.47` |
| Bot location | `/root/kingbot/` |
| Config file | `/root/kingbot/.env` |
| Service name | `kingbot` |

---

## Edit API Keys

If you need to change your Bybit or Telegram keys:

1. Connect to VPS
2. Run: `nano /root/kingbot/.env`
3. Make changes
4. Save: `Ctrl+O` → `Enter` → `Ctrl+X`
5. Restart bot: `systemctl restart kingbot`

---

## Troubleshooting

### Bot not starting?
Check the logs for errors:
```bash
journalctl -u kingbot -n 100
```

### Need to update the bot?
```bash
cd /root/kingbot
git pull
systemctl restart kingbot
```

### VPS not responding?
Go to DigitalOcean dashboard and use the web console, or reboot the droplet.

---

## Quick Reference Card

```
Connect:     ssh root@209.38.84.47
Logs:        journalctl -u kingbot -f
Status:      systemctl status kingbot
Stop:        systemctl stop kingbot
Start:       systemctl start kingbot
Restart:     systemctl restart kingbot
Disconnect:  exit
```
