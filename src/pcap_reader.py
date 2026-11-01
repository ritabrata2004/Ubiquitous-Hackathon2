"""Read Nexmon CSI UDP packets from a .pcap capture.

Nexmon sends each CSI observation as a UDP datagram (port 5500) whose payload is an
18-byte header followed by the raw CSI words:

    offset  size  field
    0       2     magic (0x1111)
    2       1     RSSI (int8)
    3       1     frame control byte
    4       6     source MAC
    10      2     sequence number (65535 when the frame carries none)
    12      2     core / spatial-stream
    14      2     chanspec
    16      2     chip version
    18      4*N   CSI, one little-endian uint32 per subcarrier
"""
import struct

import dpkt
import numpy as np

HEADER = struct.Struct("<HbB6sHHHH")


def read_csi_packets(path, udp_port=5500, magic=0x1111, n_subcarriers=256):
    """Return (words, stats) in capture record order.

    words: uint32 array (n_packets, n_subcarriers) of packed CSI.
    stats: counts of packets seen and rejected as malformed.
    """
    payload_len = HEADER.size + 4 * n_subcarriers
    words, stats = [], {"packets": 0, "not_csi_udp": 0, "bad_magic": 0, "bad_length": 0}
    with open(path, "rb") as fh:
        for _, buf in dpkt.pcap.Reader(fh):
            stats["packets"] += 1
            eth = dpkt.ethernet.Ethernet(buf)
            udp = getattr(eth.data, "data", None)
            if not isinstance(udp, dpkt.udp.UDP) or udp.dport != udp_port:
                stats["not_csi_udp"] += 1
                continue
            p = bytes(udp.data)
            if len(p) != payload_len:
                stats["bad_length"] += 1
                continue
            if HEADER.unpack_from(p)[0] != magic:
                stats["bad_magic"] += 1
                continue
            words.append(np.frombuffer(p, dtype="<u4", offset=HEADER.size))
    words = np.array(words, dtype=np.uint32).reshape(-1, n_subcarriers)
    return words, stats
