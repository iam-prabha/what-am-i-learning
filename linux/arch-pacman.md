# 🐧 Arch Linux Pacman Command Cheat Sheet

<!--toc:start-->
- [🐧 Arch Linux Pacman Command Cheat Sheet](#🐧-arch-linux-pacman-command-cheat-sheet)
  - [📦 Basic Package Operations](#📦-basic-package-operations)
  - [🔍 Searching and Querying](#🔍-searching-and-querying)
  - [🧹 Cache and Cleanup](#🧹-cache-and-cleanup)
  - [📁 Database and File Integrity](#📁-database-and-file-integrity)
  - [🧰 Advanced Operations](#🧰-advanced-operations)
  - [🧱 Package Groups Examples](#🧱-package-groups-examples)
  - [🪄 Tips and Tricks](#🪄-tips-and-tricks)
  - [🔧 AUR (Arch User Repository) Helpers](#🔧-aur-arch-user-repository-helpers)
  - [🧾 Logs](#🧾-logs)
  - [⚡ Common Fixes](#common-fixes)
  - [📘 References](#📘-references)
<!--toc:end-->

## 📦 Basic Package Operations

| Action | Command | Description |
|--------|----------|-------------|
| **Update package database and system** | `sudo pacman -Syu` | Sync repos and upgrade all packages |
| **Update only package database** | `sudo pacman -Sy` | Update package database only |
| **Upgrade system (safe)** | `sudo pacman -Syu` | Recommended regular update |
| **Install package** | `sudo pacman -S <package>` | Install a new package |
| **Reinstall package** | `sudo pacman -S <package> --overwrite '*'` | Reinstall package |
| **Remove package** | `sudo pacman -R <package>` | Remove package (keep deps) |
| **Remove package and unneeded deps** | `sudo pacman -Rs <package>` | Remove and clean unused deps |
| **Remove package, configs, and deps** | `sudo pacman -Rns <package>` | Deep clean uninstall |
| **Install a local package (.pkg.tar.zst)** | `sudo pacman -U <path-to-pkg>` | Install manually downloaded package |
| **Install from URL** | `sudo pacman -U https://...` | Install remote package |

## 🔍 Searching and Querying

| Action | Command | Description |
|--------|----------|-------------|
| **Search package in repos** | `pacman -Ss <keyword>` | Search repo packages |
| **Search installed packages** | `pacman -Qs <keyword>` | Search installed packages |
| **List all installed packages** | `pacman -Q` | Show installed packages |
| **List explicitly installed packages** | `pacman -Qe` | Show user-installed packages |
| **List dependencies** | `pacman -Qd` | Show dependency packages |
| **List orphaned packages** | `pacman -Qdt` | Show packages no longer needed |
| **List files owned by package** | `pacman -Ql <package>` | List package contents |
| **Find which package owns a file** | `pacman -Qo /path/to/file` | Reverse file lookup |
| **Show detailed info for a package** | `pacman -Qi <package>` | Installed package info |
| **Show info for repo package** | `pacman -Si <package>` | Repo package info |

## 🧹 Cache and Cleanup

| Action | Command | Description |
|--------|----------|-------------|
| **Remove all cached packages except latest 3** | `sudo paccache -r` | Clean old package versions |
| **Remove all cached packages** | `sudo paccache -rk0` | Clean entire cache |
| **Remove uninstalled package cache** | `sudo pacman -Sc` | Clean old cache entries |
| **Clean all cache completely** | `sudo pacman -Scc` | Full clean (ask confirmation) |
| **Remove unused dependencies** | `sudo pacman -Rns $(pacman -Qdtq)` | Clean orphaned deps |

## 📁 Database and File Integrity

| Action | Command | Description |
|--------|----------|-------------|
| **Verify installed packages** | `sudo pacman -Qk` | Check package integrity |
| **Verify a specific package** | `sudo pacman -Qkk <package>` | Verify all package files |
| **Force refresh of all package databases** | `sudo pacman -Syy` | Double `y` forces refresh |
| **Fix database issues** | `sudo pacman -D --asdeps <pkg>` | Mark package as dependency |
| **Rebuild package database** | `sudo pacman -D --asexplicit <pkg>` | Mark package as explicitly installed |

## 🧰 Advanced Operations

| Action | Command | Description |
|--------|----------|-------------|
| **Show package changelog** | `pacman -Qc <package>` | Show changelog (if available) |
| **Download package without installing** | `pacman -Sw <package>` | Save package to cache only |
| **Sync all mirrors and update** | `sudo reflector --latest 10 --sort rate --save /etc/pacman.d/mirrorlist` | Update mirrorlist |
| **List all repo groups** | `pacman -Sg` | Show groups available |
| **Install group** | `sudo pacman -S <group>` | Install full group |
| **List packages in group** | `pacman -Sg <group>` | Show group packages |

## 🧱 Package Groups Examples

| Group | Command |
|-------|----------|
| Base Development | `sudo pacman -S base-devel` |
| Xorg Display Server | `sudo pacman -S xorg` |
| KDE Plasma | `sudo pacman -S plasma kde-applications` |
| GNOME | `sudo pacman -S gnome gnome-extra` |
| Fonts | `sudo pacman -S ttf-dejavu ttf-liberation noto-fonts` |

## 🪄 Tips and Tricks

```bash
# List recently installed packages
grep "\[ALPM\] installed" /var/log/pacman.log

# Backup installed packages list
pacman -Qqe > pkglist.txt

# Restore from list
sudo pacman -S --needed - < pkglist.txt

# Check disk space of installed packages
expac -H M "%-30n %m" | sort -hk2
```

## 🔧 AUR (Arch User Repository) Helpers

| Tool | Command | Description |
|------|----------|-------------|
| **Install yay (recommended)** | `sudo pacman -S --needed git base-devel && git clone https://aur.archlinux.org/yay.git && cd yay && makepkg -si` | Install yay |
| **Search in AUR** | `yay -Ss <package>` | Search AUR |
| **Install from AUR** | `yay -S <package>` | Install AUR package |
| **Update all (AUR + Pacman)** | `yay -Syu` | System + AUR update |
| **Remove with config** | `yay -Rns <package>` | Remove package completely |
| **Clean AUR cache** | `yay -Sc` | Clean yay cache |

## 🧾 Logs

| Action | Command | Description |
|--------|----------|-------------|
| **View pacman log** | `cat /var/log/pacman.log` | Display all package actions |
| **Follow log updates live** | `tail -f /var/log/pacman.log` | Real-time log view |

## ⚡ Common Fixes

| Issue | Fix |
|--------|------|
| Database locked | `sudo rm /var/lib/pacman/db.lck` |
| Mirror timeout | Use `reflector` to update mirrors |
| Keyring issues | `sudo pacman-key --init && sudo pacman-key --populate archlinux` |
| Broken packages | `sudo pacman -Qk` then reinstall missing ones |

## 📘 References

- [Arch Linux Pacman Manual](https://man.archlinux.org/man/pacman.8)
- [Arch Wiki: Pacman Tips](https://wiki.archlinux.org/title/pacman)
- [Arch Wiki: Package Management](https://wiki.archlinux.org/title/Package_management)
