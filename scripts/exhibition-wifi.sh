#!/usr/bin/env bash
# Isolate the Wi-Fi hotspot from the LAN to simulate the exhibition setup.
#
#   sudo scripts/exhibition-wifi.sh on       # cut Wi-Fi -> LAN/Internet/VPN
#   sudo scripts/exhibition-wifi.sh off      # restore normal routing
#   sudo scripts/exhibition-wifi.sh status   # show current state
#
# Adds a separate nftables table (inet laya_exhibit), so it never touches the
# hotspot's own iptables rules or Docker's chains. Removing the table restores
# everything exactly as before. Not persisted: a reboot also turns it off.
#
# Hotspot clients can still reach the Pi itself and the Docker services
# (snake :8080, showcase :8091, API :8001), since those are forwarded to the
# Docker bridges, which stay allowed.
set -euo pipefail

TABLE=laya_exhibit
WIFI_IF=${WIFI_IF:-wlan0}
UPLINK_IFS=${UPLINK_IFS:-eth0, wg0}

on() {
    nft delete table inet "$TABLE" 2>/dev/null || true
    nft -f - <<NFT
table inet $TABLE {
    chain forward {
        # Runs before the iptables-nft filter chains; a drop here is final.
        type filter hook forward priority filter - 10; policy accept;
        # Wi-Fi clients may only be forwarded into the Docker bridges
        # ("accept" here just hands the packet on to Docker's own rules).
        iifname "$WIFI_IF" oifname "br-*" accept
        iifname "$WIFI_IF" oifname "docker0" accept
        iifname "$WIFI_IF" counter drop
        # No new connections from the LAN/VPN into the Wi-Fi clients.
        iifname { $UPLINK_IFS } oifname "$WIFI_IF" ct state new counter drop
    }
}
NFT
    echo "Exhibition mode ON: $WIFI_IF isolated from { $UPLINK_IFS }."
}

off() {
    if nft delete table inet "$TABLE" 2>/dev/null; then
        echo "Exhibition mode OFF: normal routing restored."
    else
        echo "Exhibition mode was already off."
    fi
}

status() {
    if nft list table inet "$TABLE" 2>/dev/null; then
        echo "Exhibition mode: ON"
    else
        echo "Exhibition mode: OFF"
    fi
}

if [[ $EUID -ne 0 ]]; then
    echo "Run as root (sudo)." >&2
    exit 1
fi

case "${1:-}" in
    on) on ;;
    off) off ;;
    status) status ;;
    *) echo "Usage: $0 on|off|status" >&2; exit 2 ;;
esac
