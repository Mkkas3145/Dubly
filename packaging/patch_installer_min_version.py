"""Change the Inno 7.1 installer OS gate without changing companion BINs.

Preserves every file location, data checksum and payload byte from the original
release. Rebuilds only the setup metadata block and its documented CRC tables.
"""
from pathlib import Path
import argparse,lzma,struct,zlib

MAGIC=b'rDlPtS\xcd\xe6\xd7\x7b\x0b\x2a'
TABLE='<12sIqqIIqqII'

def decode_block(data,offset):
    checksum,size,compressed=struct.unpack_from('<IqB',data,offset)
    assert checksum==zlib.crc32(data[offset+4:offset+13]),'Invalid block header CRC'
    remaining=data[offset+13:offset+13+size];pieces=[]
    while remaining:
        checksum,=struct.unpack_from('<I',remaining)
        chunk=remaining[4:4100]
        assert checksum==zlib.crc32(chunk),'Invalid compressed chunk CRC'
        pieces.append(chunk);remaining=remaining[4+len(chunk):]
    content=b''.join(pieces)
    if compressed:
        prop=content[0];lc=prop%9;prop//=9;lp=prop%5;pb=prop//5
        filters=[{'id':lzma.FILTER_LZMA1,'dict_size':struct.unpack_from('<I',content,1)[0],'lc':lc,'lp':lp,'pb':pb}]
        raw=lzma.decompress(content[5:],format=lzma.FORMAT_RAW,filters=filters)
    else:raw=content;filters=None
    return raw,13+size,filters

def patch(source,destination,build):
    data=Path(source).read_bytes();positions=[];at=data.find(MAGIC)
    while at>=0:
        if at+64<=len(data) and zlib.crc32(data[at:at+60])==struct.unpack_from('<I',data,at+60)[0]:positions.append(at)
        at=data.find(MAGIC,at+1)
    assert len(positions)==1,'Expected one Inno offset table'
    table_at=positions[0];table=list(struct.unpack_from(TABLE,data,table_at))
    assert table[1]==2 and table[7]==0 and table[2]==len(data),'Unsupported installer layout'
    setup=table[6];assert data[setup:].startswith(b'Inno Setup Setup Data (7.0.0.3)')
    # SetupID 64 + encryption header CRC 4 + packed encryption header 49.
    assert data[setup+68]==0,'Encrypted installers are not supported'
    block=setup+117
    raw,old_size,filters=decode_block(data,block)
    p=0
    # 39 Unicode fields and 4 ANSI fields precede the numeric setup header.
    for _ in range(43):
        n,=struct.unpack_from('<I',raw,p);p+=4
        assert n<=len(raw)-p,'Invalid setup string length'
        p+=n
    field=p+18*4
    assert struct.unpack_from('<IIH',raw,field)==(0,(10<<24)|22000,0),'Unexpected original Windows requirement'
    updated=bytearray(raw);struct.pack_into('<I',updated,field+4,(10<<24)|build)
    properties=bytes([(filters[0]['pb']*5+filters[0]['lp'])*9+filters[0]['lc']])+struct.pack('<I',filters[0]['dict_size'])
    encoded=properties+lzma.compress(bytes(updated),format=lzma.FORMAT_RAW,filters=filters)
    chunks=b''.join(struct.pack('<I',zlib.crc32(encoded[i:i+4096]))+encoded[i:i+4096] for i in range(0,len(encoded),4096))
    header=struct.pack('<qB',len(chunks),1)
    replacement=struct.pack('<I',zlib.crc32(header))+header+chunks
    result=bytearray(data[:block]+replacement+data[block+old_size:])
    delta=len(replacement)-old_size;table[2]+=delta
    for i in (3,6,7):
        if table[i]>=block+old_size:table[i]+=delta
    table[-1]=0;packed=struct.pack(TABLE,*table);table[-1]=zlib.crc32(packed[:60])
    result[table_at:table_at+64]=struct.pack(TABLE,*table)
    # Exact metadata comparison ensures no file list, location or data hash changed.
    checked,_,_=decode_block(result,block)
    assert checked==bytes(updated) and len(result)==table[2]
    assert result[table[3]:]==data[struct.unpack_from(TABLE,data,table_at)[3]:]
    Path(destination).parent.mkdir(parents=True,exist_ok=True);Path(destination).write_bytes(result)
    print('Installer OS requirement changed to Windows 10 build',build)
    print('All original companion BIN metadata and payload checksums preserved.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('destination');p.add_argument('--build',type=int,default=17763);a=p.parse_args();patch(a.source,a.destination,a.build)
