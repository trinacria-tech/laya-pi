# Exhibition mode: isolate the Wi-Fi hotspot

At the exhibition the Pi's hotspot (`trinacria-pi-hotspot`, `wlan0`,
`192.168.42.0/24`) will have no uplink. At home it normally routes to the LAN
(`eth0`) and the Internet. Exhibition mode cuts that routing so you can test
what visitors will actually get.

## Commands

```bash
sudo scripts/exhibition-wifi.sh on       # isolate Wi-Fi from LAN / Internet / VPN
sudo scripts/exhibition-wifi.sh status   # show state and drop counters
sudo scripts/exhibition-wifi.sh off      # restore normal routing
```

The rule is **not persistent**: a reboot turns it off too.

## What still works with it on

Hotspot clients can still reach:

- the Pi itself (`192.168.42.1`: DHCP, DNS, SSH)
- the demo services published by Docker:
  - snake: `http://192.168.42.1:8080`
  - showcase: `http://192.168.42.1:8091`
  - API: `http://192.168.42.1:8001`

What gets blocked:

- Wi-Fi → LAN (`eth0`), Internet and WireGuard (`wg0`)
- new connections from the LAN/VPN to Wi-Fi clients

## How it works

The script adds its own nftables table, `inet laya_exhibit`, with a `forward`
chain at priority `filter - 10`. That runs before the hotspot's iptables rules
and Docker's chains, so its drop is final. Traffic from `wlan0` going to the
Docker bridges (`br-*`, `docker0`) is let through. Everything else from `wlan0`
is dropped.

It changes nothing else: the hotspot's iptables rules, NAT and Docker all stay
as they are. `off` just deletes the table.

You can change the interfaces with environment variables:

```bash
sudo WIFI_IF=wlan0 UPLINK_IFS="eth0, wg0" scripts/exhibition-wifi.sh on
```

## Quick check from a phone on the hotspot

1. `on`, then from the phone open `http://192.168.42.1:8080`: the snake loads.
2. Try opening a website or a LAN address (e.g. `192.168.87.1`): it fails.
3. `status` on the Pi: the `drop` counter goes up.
4. `off`: browsing works again.

Note: the phone may say "no Internet connection" and switch to mobile data.
Turn mobile data off during the test, just as it would be at the exhibition if
there's no signal.
