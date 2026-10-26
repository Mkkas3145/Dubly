"""Read common Qwen text-repair formats without guessing dialogue identity."""
import ast,json,re

def unique(pairs):
    output={}
    for key,value in pairs:
        if key in output:raise ValueError('duplicate_response_key')
        output[key]=value
    return output

def literal_container(text):
    # Accept single quotes/trailing commas as literal data, never executable
    # expressions. Explicitly reject duplicate keys before literal_eval.
    tree=ast.parse(text,mode='eval')
    for node in ast.walk(tree):
        if isinstance(node,ast.Dict):
            keys=[ast.literal_eval(key) for key in node.keys]
            if len(set(keys))!=len(keys):raise ValueError('duplicate_response_key')
    result=ast.literal_eval(tree)
    if not isinstance(result,(dict,list)):raise ValueError('not_a_response_container')
    return result

def region_id(value):
    if type(value) is int and value>=0:return value
    if isinstance(value,str) and re.fullmatch(r'[0-9]+',value):return int(value)
    raise ValueError('invalid_region_id')

def normalize(value,expected_ids=None):
    expected=None if expected_ids is None else list(expected_ids)
    for _ in range(4):
        if not isinstance(value,dict):break
        if value and all(re.fullmatch(r'[0-9]+',str(k)) for k in value):
            entries=[]
            for key,item in value.items():
                if isinstance(item,str):item={'id':key,'text':item}
                elif isinstance(item,dict):
                    item=dict(item)
                    if 'id' in item and region_id(item['id'])!=region_id(key):raise ValueError('conflicting_region_ids')
                    item['id']=key
                else:raise ValueError('invalid_region_text')
                entries.append(item)
            value=entries;break
        wrappers=[key for key in ('segments','corrections','results','lines','data','result','output') if isinstance(value.get(key),(list,dict))]
        if len(wrappers)!=1:raise ValueError('ambiguous_response_container')
        value=value[wrappers[0]]
    if not isinstance(value,list):raise ValueError('invalid_punctuation_response')
    if value and all(isinstance(item,str) for item in value):
        if expected is None or len(value)!=len(expected):raise ValueError('unidentified_region_text')
        value=[{'id':i,'text':text} for i,text in zip(expected,value)]
    entries=[];seen=set()
    for item in value:
        if isinstance(item,list) and len(item)==2:item={'id':item[0],'text':item[1]}
        if not isinstance(item,dict):raise ValueError('invalid_punctuation_segment')
        ids=[region_id(item[key]) for key in ('id','region_id','segment_id','index') if key in item]
        if not ids or len(set(ids))!=1:raise ValueError('invalid_or_conflicting_region_id')
        identifier=ids[0]
        text=next((item[key] for key in ('corrected_text','punctuated_text','corrected','text','sentence') if isinstance(item.get(key),str)),None)
        if text is None or identifier in seen:raise ValueError('invalid_or_duplicate_region_text')
        seen.add(identifier);entries.append({'id':identifier,'text':text})
    if expected is not None:
        if seen!=set(expected):raise ValueError('region_ids_changed')
        by_id={entry['id']:entry for entry in entries};entries=[by_id[i] for i in expected]
    return {'segments':entries}

def parse(content,expected_ids):
    content=content.strip().lstrip('\ufeff')
    try:value=json.loads(content,object_pairs_hook=unique)
    except json.JSONDecodeError:
        if content.startswith(('{','[')):
            try:value=literal_container(content)
            except (ValueError,SyntaxError,TypeError):pass
            else:return normalize(value,expected_ids)
    else:return normalize(value,expected_ids)
    candidates=[]
    fenced=re.findall(r'```(?:json)?\s*([\s\S]*?)```',content,flags=re.I)
    decoder=json.JSONDecoder(object_pairs_hook=unique)
    for source in fenced or [content]:
        try:value=literal_container(source.strip())
        except (ValueError,SyntaxError,TypeError):pass
        else:
            candidates.append(normalize(value,expected_ids));continue
        # Decode complete outer JSON blocks, skipping their nested openers.
        cursor=0
        while cursor<len(source):
            match=re.search(r'[\[{]',source[cursor:])
            if not match:break
            start=cursor+match.start()
            try:value,end=decoder.raw_decode(source,start)
            except json.JSONDecodeError:cursor=start+1;continue
            cursor=end
            try:candidates.append(normalize(value,expected_ids))
            except ValueError:continue
    if candidates:
        if any(candidate!=candidates[0] for candidate in candidates[1:]):raise ValueError('ambiguous_response_blocks')
        return candidates[0]
    entries=[]
    for line in content.splitlines():
        match=re.fullmatch(r'\s*(?:[-*]\s*)?\[?([0-9]+)\]?\s*[:：\t]\s*(.+?)\s*',line)
        if match:entries.append({'id':match[1],'text':match[2]})
    if entries:return normalize(entries,expected_ids)
    raise ValueError('unrecognized_punctuation_response')
