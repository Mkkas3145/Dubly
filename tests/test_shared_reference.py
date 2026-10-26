import io,json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import soundfile as sf
import render
import shared_reference as shared
from core import SR,config

def fixture(tmp_path):
    clean=tmp_path/'clean.wav'
    t=np.arange(3*SR)/SR
    sf.write(clean,np.repeat((.1*np.sin(2*np.pi*220*t))[:,None],2,axis=1),SR,subtype='FLOAT')
    segments=[{'start':float(i),'end':i+.25,'lo':i*SR,'hi':i*SR+SR//4,
        'original_text':f'原文{i}','translated_text':f'번역{i}'} for i in range(3)]
    timeline={'turns':[{**s,'speaker':['A','B','A'][i]} for i,s in enumerate(segments)]}
    return clean,segments,timeline

def test_plan_never_shares_different_or_uncertain_voices_and_caps_16():
    segments=[{'start':float(i),'end':i+.5,'original_text':'原文','translated_text':'번역'} for i in range(38)]
    turns=[{**s,'speaker':'A' if i%2==0 else 'B'} for i,s in enumerate(segments)]
    groups=shared.plan(segments,{'turns':turns})
    assert [len(g['indices']) for g in groups]==[16,16,3,3]
    for g in groups:assert {turns[i]['speaker'] for i in g['indices']}=={g['speaker']}
    turns.append({'start':0.,'end':.5,'speaker':'B'})
    turns[1]['speaker']='unknown'
    groups=shared.plan(segments,{'turns':turns})
    assert next(g for g in groups if 0 in g['indices'])=={'speaker':None,'indices':[0],'reference_indices':[0]}
    assert next(g for g in groups if 1 in g['indices'])=={'speaker':None,'indices':[1],'reference_indices':[1]}

def test_remainder_uses_16_references_without_duplicating_output_requests():
    for count in range(1,66):
        segments=[{'start':float(i),'end':i+.2,'original_text':str(i),'translated_text':str(i)} for i in range(count)]
        turns=[{**s,'speaker':'A'} for s in segments]
        groups=shared.plan(segments,{'turns':turns})
        assert [i for g in groups for i in g['indices']]==list(range(count))
        for g in groups:
            assert len(g['reference_indices'])==min(16,count)
            assert len(set(g['reference_indices']))==len(g['reference_indices'])
            assert set(g['indices'])<=set(g['reference_indices'])
        if count==17:
            assert groups[0]['reference_indices']==list(range(16))
            assert groups[1]['indices']==[16]
            assert groups[1]['reference_indices']==list(range(1,17))

def test_last_single_output_receives_full_shared_bank(tmp_path):
    segments=[{'start':float(i),'end':i+.2,'lo':i*SR,'hi':i*SR+10,
        'original_text':f'原文{i}','translated_text':f'번역{i}'} for i in range(17)]
    timeline={'turns':[{**s,'speaker':'A'} for s in segments]}
    def crop(clean,segment,path):
        sf.write(path,np.full(10,int(segment['start'])+1,dtype=np.int16),SR,subtype='PCM_16')
    with patch.object(render,'reference_audio',side_effect=crop):
        result=shared.prepare(segments,None,tmp_path,timeline)
    assert result['individual_outputs']==17 and result['shared_groups']==2
    assert segments[16]['_shared_reference_indices']==[16,*range(1,16)]
    assert segments[16]['_shared_reference_path']!=segments[0]['_shared_reference_path']
    audio,_=sf.read(segments[16]['_shared_reference_path'],dtype='int16')
    np.testing.assert_array_equal(audio,np.repeat(np.array([17,*range(2,17)],dtype=np.int16),10))
    assert segments[16]['_shared_reference_text']=='原文16\n'+'\n'.join(f'原文{i}' for i in range(1,16))
    assert segments[16]['translated_text']=='번역16'

def test_shared_audio_is_exact_concatenation_and_placement_is_unchanged(tmp_path):
    clean,segments,timeline=fixture(tmp_path)
    result=shared.prepare(segments,clean,tmp_path,timeline)
    assert result['shared_groups']==1 and result['individual_outputs']==3
    assert segments[0]['_shared_reference_path']!=segments[2]['_shared_reference_path']
    assert '_shared_reference_path' not in segments[1]
    assert segments[0]['_shared_reference_text']=='原文0\n原文2'
    assert segments[2]['_shared_reference_text']=='原文2\n原文0'
    assert [(s['lo'],s['hi']) for s in segments]==[(i*SR,i*SR+SR//4) for i in range(3)]
    expected=np.concatenate([sf.read(tmp_path/f'{i:04d}-reference.wav',dtype='int16')[0] for i in (0,2)])
    actual,_=sf.read(segments[0]['_shared_reference_path'],dtype='int16')
    np.testing.assert_array_equal(actual,expected)
    second,_=sf.read(segments[2]['_shared_reference_path'],dtype='int16')
    np.testing.assert_array_equal(second,np.concatenate([sf.read(tmp_path/f'{i:04d}-reference.wav',dtype='int16')[0] for i in (2,0)]))

def test_every_request_moves_only_its_own_reference_to_front():
    for count in range(1,17):
        members=list(range(count))
        for target in members:
            order=shared.ordered_indices(members,target)
            assert order[0]==target and len(order)==count
            assert len(set(order))==count and set(order)==set(members)
            assert order[1:]==[i for i in members if i!=target]

def test_preview_keeps_full_donor_group_while_building_only_selected_request(tmp_path):
    clean,segments,timeline=fixture(tmp_path)
    result=shared.prepare(segments,clean,tmp_path,timeline,only_indices=[2])
    assert result['individual_outputs']==1
    assert '_shared_reference_path' not in segments[0]
    assert segments[2]['_shared_reference_indices']==[2,0]
    assert segments[2]['_shared_reference_text']=='原文2\n原文0'

def test_failed_member_reference_falls_back_without_poisoning_group(tmp_path):
    clean,segments,timeline=fixture(tmp_path)
    original=render.reference_audio
    def crop(clean,segment,destination):
        if segment['lo']==2*SR:raise RuntimeError('missing voice')
        return original(clean,segment,destination)
    with patch.object(render,'reference_audio',side_effect=crop):
        result=shared.prepare(segments,clean,tmp_path,timeline)
    assert result['shared_groups']==0
    assert all('_shared_reference_path' not in s for s in segments)
    assert any(g.get('reason')=='unusable_member_reference' for g in result['groups'])

def test_generation_uses_shared_reference_but_timing_uses_own_clip_and_cache_keys(tmp_path):
    clean,segments,timeline=fixture(tmp_path)
    shared.prepare(segments,clean,tmp_path,timeline)
    target={**segments[0],'next_dialogue_start_frame':SR}
    seen=[]
    def request(reference,segment,language,key,destination):
        seen.append((reference,segment.copy()));destination.write_bytes(b'test')
        return {'generated_seconds':.4}
    with patch.object(render,'request_audio',side_effect=request):
        prepared=render.generate_speech(0,target,clean,tmp_path,{'language':'ko','key':'private'})
        render.generate_speech(0,target,clean,tmp_path,{'language':'ko','key':'private'})
        assert len(seen)==1
        render.generate_speech(0,{**target,'_shared_reference_text':'changed'},clean,tmp_path,{'language':'ko','key':'private'})
        assert len(seen)==2
    assert seen[0][0]==Path(target['_shared_reference_path'])
    assert prepared['timing_reference']==tmp_path/'0000-reference.wav'
    assert sf.info(prepared['timing_reference']).frames==SR//4
    assert prepared['frames']==SR//4 and prepared['collision_budget']==SR
    assert prepared['segment']['original_text']=='原文0'

def test_api_payload_removes_stale_speaker_tags_but_keeps_reference_text(tmp_path):
    reference=tmp_path/'shared.wav';reference.write_bytes(b'fixed audio')
    seen=[]
    class Response(io.BytesIO):headers={}
    def reply(request,**kwargs):seen.append(json.loads(request.data));return Response(b'audio')
    segment={'original_text':'原文0','translated_text':'번역0','_shared_reference_text':'<|speaker:0|>原文0 <|speaker:1|>原文2','_shared_reference_tagged':True,'_shared_output_slot':0}
    with patch.object(render.urllib.request,'urlopen',side_effect=reply),patch.object(render,'probe',return_value={'format':{'duration':'1'}}):
        render.request_audio(reference,segment,'ko','private',tmp_path/'one.mp3')
        render.request_audio(reference,{**segment,'translated_text':'번역2','_shared_output_slot':1},'ko','private',tmp_path/'two.mp3')
    assert [p['input'] for p in seen]==['번역0','번역2']
    assert seen[0]['input_references']==seen[1]['input_references']
    assert seen[0]['input_references'][1]['text']=='原文0 原文2'
    assert '<|speaker:' not in json.dumps(seen)


def test_donor_marker_is_preserved_only_in_reference(tmp_path):
    reference=tmp_path/'reference.wav';reference.write_bytes(b'audio')
    seen=[]
    class Response(io.BytesIO):headers={}
    def reply(request,**kwargs):seen.append(json.loads(request.data));return Response(b'audio')
    segment={'original_text':'原文0','translated_text':'<|speaker:0|>번역0',
        '_shared_reference_text':'原文0 <|speaker:1|>原文1 原文2','_shared_reference_donor_tag':True}
    with patch.object(render.urllib.request,'urlopen',side_effect=reply),patch.object(render,'probe',return_value={'format':{'duration':'1'}}):
        render.request_audio(reference,segment,'ko','private',tmp_path/'one.mp3')
    payload=seen[0]
    assert payload['input']=='번역0'
    assert payload['input_references'][1]['text']=='原文0 <|speaker:1|>原文1 原文2'
    assert '<|speaker:0|>' not in json.dumps(payload)
    assert '<|speaker:1|>' in payload['input_references'][1]['text']


def test_context_reference_target_first_marker_reaches_actual_request(tmp_path):
    from batch_tts import prepare_context_references
    import numpy as np
    import soundfile as sf
    clean=tmp_path/'clean.wav';sf.write(clean,np.repeat([.1,.2,.3],44100),44100)
    rows=[{'start':float(i),'end':float(i+1),'lo':i*44100,'hi':(i+1)*44100,
           'speaker_id':'0','original_text':f'原文{i}','translated_text':f'번역{i}'} for i in range(3)]
    prepare_context_references(rows,clean,tmp_path,only_indices=[2])
    seen=[]
    class Response(io.BytesIO):headers={}
    def reply(request,**kwargs):seen.append(json.loads(request.data));return Response(b'audio')
    with patch.object(render.urllib.request,'urlopen',side_effect=reply),patch.object(render,'probe',return_value={'format':{'duration':'1'}}):
        render.request_audio(Path(rows[2]['_shared_reference_path']),rows[2],'ko','private',tmp_path/'result.mp3')
    assert seen[0]['input']=='번역2'
    assert 'instructions' not in seen[0]
    assert seen[0]['input_references'][1]['text']=='原文2\n原文1'
    assert '<|speaker:' not in json.dumps(seen[0])
    assert rows[2]['_shared_reference_indices']==[2,1]


def test_acoustic_selection_can_cross_batch_boundary_without_extra_audio_or_text(tmp_path):
    segments=[{'start':float(i),'end':i+.2,'lo':i*SR,'hi':i*SR+10,'speaker_id':'A',
        'original_text':f'原文{i}','translated_text':f'번역{i}'} for i in range(17)]
    segments[16]['_reference_donor_indices']=[16,0,2]
    timeline={'turns':[{**s,'speaker':'A'} for s in segments]}
    def crop(clean,s,path):sf.write(path,np.full(10,int(s['start'])+1,dtype=np.int16),SR,subtype='PCM_16')
    with patch.object(render,'reference_audio',side_effect=crop):shared.prepare(segments,None,tmp_path,timeline,only_indices=[16])
    audio,_=sf.read(segments[16]['_shared_reference_path'],dtype='int16')
    np.testing.assert_array_equal(audio,np.repeat(np.array([17,1,3],dtype=np.int16),10))
    assert segments[16]['_shared_reference_indices']==[16,0,2]
    assert segments[16]['_shared_reference_text']=='原文16\n原文0\n原文2'
