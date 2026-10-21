"""Unchanged scalar decoder helpers from the existing CPU proof."""
import struct

def windows_bits(tile,bits):
    """Independent cyclic, MSB-first bitstream definition, tile-local wrap."""
    words=struct.unpack('<'+'I'*(len(tile)//4),tile)
    stream=''.join(f'{w:032b}' for w in words)
    out=[]
    for t in range(256):
        end=(t+1)*bits
        out.append(int(''.join(stream[i%len(stream)] for i in range(end-16,end)),2))
    return out

def half_bits(v):return struct.unpack('<H',struct.pack('<e',v))[0]

def mul1(index):
    x=(index*0x83DCD12D)&0xffffffff
    sm=sum(x>>j&255 for j in (0,8,16,24))
    a=struct.unpack('<e',struct.pack('<H',0x6400+sm))[0]
    inv=struct.unpack('<e',struct.pack('<H',0x1eee))[0]
    bias=struct.unpack('<e',struct.pack('<H',0xc931))[0]
    # Binary64 represents this half*half + half exactly before one FP16 rounding.
    return half_bits(a*inv+bias)
