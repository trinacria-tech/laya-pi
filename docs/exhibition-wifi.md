# Modalità fiera: isolare l'hotspot Wi-Fi

In fiera l'hotspot del Pi (`trinacria-pi-hotspot`, `wlan0`, `192.168.42.0/24`)
non avrà uscita verso l'esterno. A casa invece di solito instrada verso la LAN
(`eth0`) e Internet. La modalità fiera taglia questo instradamento, così puoi
provare cosa troveranno davvero i visitatori.

## Comandi

```bash
sudo scripts/exhibition-wifi.sh on       # isola il Wi-Fi da LAN / Internet / VPN
sudo scripts/exhibition-wifi.sh status   # mostra lo stato e i contatori dei pacchetti scartati
sudo scripts/exhibition-wifi.sh off      # ripristina l'instradamento normale
```

La regola **non è persistente**: anche un riavvio la disattiva.

## Cosa continua a funzionare

I client dell'hotspot possono ancora raggiungere:

- il Pi stesso (`192.168.42.1`: DHCP, DNS, SSH)
- i servizi della demo pubblicati da Docker:
  - snake: `http://192.168.42.1:8080`
  - showcase: `http://192.168.42.1:8091`
  - API: `http://192.168.42.1:8001`

Cosa viene bloccato:

- Wi-Fi → LAN (`eth0`), Internet e WireGuard (`wg0`)
- le nuove connessioni dalla LAN/VPN verso i client Wi-Fi

## Come funziona

Lo script aggiunge una sua tabella nftables, `inet laya_exhibit`, con una catena
`forward` a priorità `filter - 10`. Questa viene eseguita prima delle regole
iptables dell'hotspot e delle catene di Docker, quindi il suo drop è definitivo.
Il traffico da `wlan0` diretto ai bridge Docker (`br-*`, `docker0`) passa. Tutto
il resto proveniente da `wlan0` viene scartato.

Non modifica nient'altro: le regole iptables dell'hotspot, il NAT e Docker
restano come sono. `off` si limita a cancellare la tabella.

Puoi cambiare le interfacce con le variabili d'ambiente:

```bash
sudo WIFI_IF=wlan0 UPLINK_IFS="eth0, wg0" scripts/exhibition-wifi.sh on
```

## Verifica rapida da un telefono sull'hotspot

1. `on`, poi dal telefono apri `http://192.168.42.1:8080`: lo snake si carica.
2. Prova ad aprire un sito o un indirizzo della LAN (es. `192.168.87.1`): non funziona.
3. `status` sul Pi: il contatore del `drop` sale.
4. `off`: la navigazione torna a funzionare.

Nota: il telefono potrebbe dire "nessuna connessione a Internet" e passare ai
dati mobili. Durante la prova disattiva i dati mobili, come succederebbe in
fiera se non c'è campo.
