"""Incrementally validate JSON and count complete, valid review regions."""
import json,re


class IncrementalJSON:
    def __init__(self,completed,key_seen=lambda *a:None):
        self.completed=completed;self.key_seen=key_seen;self.text='';self.cursor=0;self.stack=[];self.root=None;self.done=False;self.started=False;self.fenced=False

    def value_path(self):
        if not self.stack:return ()
        top=self.stack[-1]
        return top['path']+((top['key'] if top['kind']=='object' else len(top['value'])),)

    def wants_value(self):
        return not self.started if not self.stack else self.stack[-1]['state'] in ('value','value_or_end')

    def attach(self,value):
        if not self.stack:self.root=value;self.done=True;return
        top=self.stack[-1]
        if top['kind']=='object':top['value'][top['key']]=value
        else:top['value'].append(value)
        top['state']='comma_or_end'

    def feed(self,chunk,final=False):
        self.text+=chunk
        if not self.started and not self.stack:
            initial=self.text.lstrip()
            if initial.startswith('`'):
                if '\n' not in initial:
                    if not final and ('```json'.startswith(initial) or '```'.startswith(initial)):return
                    raise ValueError('review_stream_invalid_json_prefix')
                opening=initial.split('\n',1)[0].strip()
                if opening not in ('```','```json','```JSON'):raise ValueError('review_stream_invalid_json_fence')
                self.cursor=len(self.text)-len(initial)+initial.index('\n')+1;self.fenced=True
        while self.cursor<len(self.text):
            c=self.text[self.cursor]
            if c.isspace():self.cursor+=1;continue
            if self.done:
                tail=self.text[self.cursor:].strip()
                if self.fenced and (tail=='```' or not final and '```'.startswith(tail)):break
                raise ValueError('review_stream_text_after_json')
            top=self.stack[-1] if self.stack else None
            if c in '}]':
                if not top or c!=('}' if top['kind']=='object' else ']') or top['state'] not in ('key_or_end','value_or_end','comma_or_end'):
                    raise ValueError('review_stream_invalid_container_end')
                self.cursor+=1;self.stack.pop();self.attach(top['value']);self.completed(top['path'],top['value']);continue
            if top and top['state']=='colon':
                if c!=':':raise ValueError('review_stream_missing_colon')
                self.cursor+=1;top['state']='value';continue
            if top and top['state']=='comma_or_end':
                if c!=',':raise ValueError('review_stream_missing_comma')
                self.cursor+=1;top['state']='key' if top['kind']=='object' else 'value';continue
            key=bool(top and top['kind']=='object' and top['state'] in ('key','key_or_end'))
            if c in '{[':
                if key or not self.wants_value():raise ValueError('review_stream_unexpected_container')
                path=self.value_path();kind='object' if c=='{' else 'array';self.started=True
                self.stack.append({'kind':kind,'value':{} if c=='{' else [],'path':path,'state':'key_or_end' if c=='{' else 'value_or_end','key':None})
                self.cursor+=1;continue
            if c=='"':
                end=self.cursor+1;escaped=False
                while end<len(self.text):
                    x=self.text[end]
                    if x=='"' and not escaped:break
                    if ord(x)<32:raise ValueError('review_stream_invalid_control_character')
                    escaped=not escaped if x=='\\' else False;end+=1
                field=self.value_path()[-1] if self.stack and not key else None
                limit=80 if field in ('emotion','delivery','affect','e') else 64 if field in ('nickname','alias','name','label') else 8192
                fragment=self.text[self.cursor+1:end]
                if len(fragment)>limit:
                    try:decoded=json.loads('"'+fragment+'"')
                    except json.JSONDecodeError:decoded=None
                    if (decoded is not None and len(decoded)>limit) or len(fragment)>limit*6+6:
                        raise ValueError('review_stream_string_too_long:'+str(field))
                if end==len(self.text):break
                value=json.loads(self.text[self.cursor:end+1]);self.cursor=end+1
                if key:
                    if value in top['value']:raise ValueError('review_stream_duplicate_field')
                    self.key_seen(top['path'],value)
                    top['key']=value;top['state']='colon'
                elif self.wants_value() and self.stack:self.attach(value)
                else:raise ValueError('review_stream_expected_object_or_array')
                continue
            if key or not self.wants_value() or not self.stack:raise ValueError('review_stream_expected_json_object_or_array')
            end=self.cursor
            while end<len(self.text) and self.text[end] not in ' \t\r\n,]}':end+=1
            token=self.text[self.cursor:end]
            if end==len(self.text) and not final:
                literal=any(v.startswith(token) for v in ('true','false','null'))
                numeric=bool(re.fullmatch(r'-?(?:\d+(?:\.\d*)?(?:[eE][+-]?\d*)?)?',token))
                if not literal and not numeric:raise ValueError('review_stream_invalid_json_token')
                break
            value=json.loads(token);self.cursor=end;self.attach(value)
        if final and (not self.done or self.stack):raise ValueError('review_stream_incomplete_json')
        return self.root


class ReviewProgress:
    def __init__(self,rows,validate,progress,*,on_region=None,completed_count=0,total=None):
        self.rows={r['id']:r for r in rows};self.validate=validate;self.progress=progress;self.seen=set()
        self.on_region=on_region;self.completed_count=completed_count;self.total=total if total is not None else len(rows)
        self.parser=IncrementalJSON(self.completed,self.key_seen)

    def key_seen(self,path,key):
        allowed={'segments','regions','results','turns','data','result','output','speaker_segments',
            'speakers','speaker_names','speaker_nicknames','nicknames','id','region_id','segment_id','index',
            'parts','splits','subsegments','first_word_id','start_word_id','word_range','w'}
        if not path and key not in allowed and not key.isdigit():
            raise ValueError('review_stream_unexpected_root_field:'+key)

    def completed(self,path,value):
        # Invalid empty parts can be rejected before the region/root finishes.
        if path and path[-1] in ('parts','splits','turns','subsegments') and value==[]:
            raise ValueError('review_stream_empty_parts')
        if not isinstance(value,dict):return
        parts=next((value[k] for k in ('parts','splits','turns','subsegments') if k in value),None)
        if parts is None:return
        identifier=next((value[k] for k in ('id','region_id','segment_id','index') if k in value),None)
        if identifier is None:return
        try:identifier=int(identifier)
        except (ValueError,TypeError):raise ValueError('review_stream_invalid_region_id') from None
        if identifier not in self.rows:raise ValueError('review_stream_unknown_region_id')
        if identifier in self.seen:raise ValueError('review_stream_duplicate_region_id')
        self.validate(self.rows[identifier],value)
        if self.on_region:self.on_region(value)
        self.seen.add(identifier)
        count,total=self.completed_count+len(self.seen),self.total
        self.progress(min(99,100*count/max(1,total)),f'감정 전환·화자 보정 응답 수신 {count}/{total}개')

    def feed(self,text):return self.parser.feed(text)
    def finish(self):return self.parser.feed('',final=True)
