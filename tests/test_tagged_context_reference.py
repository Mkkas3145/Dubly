import io,json
from unittest.mock import patch
import numpy as np
import soundfile as sf
from core import SR,config
from tagged_context_reference import prepare
import render


def rows(count=22):
    return [{'start':float(i),'end':float(i+1),'lo':i*SR,'hi':(i+1)*SR,
        'speaker_id':'A','original_text':f'source {i}','translated_text':f'target {i}'} for i in range(count)]


def test_actual_payload_keeps_reference_tags_and_instruction_without_output_tags(tmp_path):
    segments=rows();segments[20]['speaker_id']='B';segments[19]['deleted']=True
    clean=tmp_path/'clean.wav'
    values=np.concatenate([np.full((SR,1),(i+1)/32,np.float32) for i in range(len(segments))])
    sf.write(clean,values,SR,subtype='PCM_16')
    prepare(segments,clean,tmp_path,only_indices=[17])
    row=segments[17];row['_source_language']='ja';row['translated_text']='[soft] target [hesitant] 17';indices=row['_shared_reference_indices']
    assert indices[0]==17 and len(indices)==20 and len(set(indices))==20
    assert 19 not in indices and 20 not in indices
    assert indices[1:]==sorted(indices[1:])
    actual,rate=sf.read(row['_shared_reference_path'],always_2d=True)
    assert len(actual)==round(30.5*SR)
    np.testing.assert_array_equal(actual[:SR//2],0)
    np.testing.assert_array_equal(actual[-SR//2:],0)
    for slot,i in enumerate(indices):
        start=SR//2+slot*(SR+SR//2)
        np.testing.assert_allclose(actual[start:start+SR],(i+1)/32,atol=.0001)
        if slot<len(indices)-1:np.testing.assert_array_equal(actual[start+SR:start+SR+SR//2],0)
    payloads=[]
    class Reply(io.BytesIO):headers={}
    def reply(request,**kwargs):payloads.append(json.loads(request.data));return Reply(b'audio')
    with patch.object(render.urllib.request,'urlopen',side_effect=reply),patch.object(render,'probe',return_value={'format':{'duration':'1'}}):
        render.request_audio(tmp_path/'tagged-context-references'/'0017-source.wav',row,'ko','test',tmp_path/'out.mp3')
    payload=payloads[0]
    assert payload['input']=='<korean>[soft] target [hesitant] 17</korean>'
    assert row['translated_text']=='[soft] target [hesitant] 17'
    assert '_output_audio_tokens' not in row
    assert row['_shared_reference_output_tag'] is None and row['_preserve_speaker_tags'] is False
    text=payload['input_references'][1]['text']
    assert text=='<japanese>source 17</japanese> <|speaker:1|> '+' '.join(f'<japanese>source {i}</japanese>' for i in indices[1:])
    assert '<|speaker:0|>' not in text
    assert text.count('<|speaker:1|>')==1 and '<|speaker:2|>' not in text
    assert payload['instructions']==render.tts_instructions('ko',source_language='ja',structured=True)
    assert payload['provider']['options']['fish-audio']['condition_on_previous_chunks'] is True


def test_small_voice_and_unknowns_do_not_borrow_other_voices(tmp_path):
    segments=rows(4);segments[2]['speaker_id']='B';segments[3]['speaker_id']=None
    clean=tmp_path/'clean.wav';sf.write(clean,np.full((4*SR,1),.1,np.float32),SR)
    prepare(segments,clean,tmp_path)
    assert segments[0]['_shared_reference_indices']==[0,1]
    assert segments[1]['_shared_reference_indices']==[1,0]
    assert segments[2]['_shared_reference_indices']==[2]
    assert segments[3]['_shared_reference_indices']==[3]
    assert segments[0]['_shared_reference_text']=='source 0\n<|speaker:1|>source 1'
    assert segments[2]['_shared_reference_text']=='source 2'
    assert sf.info(segments[0]['_shared_reference_path']).frames==round(3.5*SR)
    assert sf.info(segments[2]['_shared_reference_path']).frames==2*SR


def test_pipeline_defaults_select_grouped_references_individual_generation():
    cfg=config()
    assert cfg['tts_mode']=='individual'
    assert cfg['tts_reference_mode']=='same_speaker_tagged_context'
    assert cfg['tts_omit_instructions'] is False


def test_reference_preparation_does_not_set_duration_hint(tmp_path):
    segments=rows(2)
    segments[0].update(start=0.,end=2.1,lo=0,hi=round(2.1*SR))
    segments[1].update(start=2.1,end=5.,lo=round(2.1*SR),hi=5*SR)
    clean=tmp_path/'clean.wav';sf.write(clean,np.full((5*SR,1),.1,np.float32),SR)
    prepare(segments,clean,tmp_path,only_indices=[0])
    assert '_output_audio_tokens' not in segments[0]
    assert sf.info(segments[0]['_shared_reference_path']).duration==6.5


def test_stale_duration_metadata_does_not_add_output_hint(tmp_path):
    reference=tmp_path/'ref.wav';reference.write_bytes(b'reference')
    segment={'original_text':'source','translated_text':'target',
        '_output_audio_tokens':45,'_output_duration_reference_seconds':2.1}
    payloads=[]
    class Reply(io.BytesIO):headers={}
    def reply(request,**kwargs):payloads.append(json.loads(request.data));return Reply(b'audio')
    with patch.object(render.urllib.request,'urlopen',side_effect=reply),patch.object(render,'probe',return_value={'format':{'duration':'2.1'}}):
        receipt=render.request_audio(reference,segment,'ko','test',tmp_path/'out.mp3')
    assert payloads[0]['input']=='target'
    assert segment['translated_text']=='target'
    assert 'output_duration_tokens' not in receipt


def test_initial_pipeline_prepares_the_same_reference_before_individual_generation(tmp_path):
    segments=rows(2);seen=[]
    def reference_prepare(values,*args):
        seen.append('reference')
        for value in values:value['_shared_reference_text']='<|speaker:0|>source'
    def individual(values,*args):
        assert all('_shared_reference_text' in value for value in values)
        seen.append('individual');return ['ok']
    with patch('tagged_context_reference.prepare',side_effect=reference_prepare),patch.object(render,'synthesize_individual_all',side_effect=individual):
        assert render.synthesize_all(segments,None,tmp_path,{**config(),'tts_reference_mode':'same_speaker_tagged_context'},lambda *a:None)==['ok']
    assert seen==['reference','individual']


def test_default_pipeline_does_not_prepare_grouped_references(tmp_path):
    segments=rows(2)
    with patch('tagged_context_reference.prepare',side_effect=AssertionError('No grouped references')),\
         patch.object(render,'synthesize_individual_all',return_value=['ok']) as individual:
        assert render.synthesize_all(segments,None,tmp_path,{**config(),'tts_reference_mode':'individual'},lambda *a:None)==['ok']
    assert individual.call_args.args[0] is segments
    assert all('_shared_reference_path' not in row for row in segments)

def test_reference_stops_after_whole_lines_reach_thirty_seconds(tmp_path):
 segments=rows(8)
 for i,row in enumerate(segments):row.update(start=i*10.,end=(i+1)*10.,lo=i*10*SR,hi=(i+1)*10*SR)
 clean=tmp_path/'clean.wav';sf.write(clean,np.full((80*SR,1),.1,np.float32),SR)
 prepare(segments,clean,tmp_path,only_indices=[4])
 row=segments[4];assert row['_shared_reference_indices']==[4,3,5]
 receipt=json.loads(__import__('pathlib').Path(row['_shared_reference_path']).with_suffix('.json').read_text('utf8'))
 assert receipt['content_seconds']==30 and receipt['reference_seconds']==32


def test_current_reference_over_thirty_seconds_needs_no_donors(tmp_path):
 segments=rows(2);segments[0].update(end=35.,hi=35*SR);segments[1].update(start=35.,end=40.,lo=35*SR,hi=40*SR)
 clean=tmp_path/'clean.wav';sf.write(clean,np.full((40*SR,1),.1,np.float32),SR)
 prepare(segments,clean,tmp_path,only_indices=[0]);assert segments[0]['_shared_reference_indices']==[0]


def test_short_references_can_exceed_sixteen_until_thirty_seconds(tmp_path):
    segments=rows(40)
    clean=tmp_path/'clean.wav';sf.write(clean,np.full((40*SR,1),.1,np.float32),SR)
    prepare(segments,clean,tmp_path,only_indices=[20])
    selected=segments[20]['_shared_reference_indices']
    assert selected[0]==20 and len(selected)==30 and len(set(selected))==30
    assert segments[20]['_shared_reference_text'].count('<|speaker:1|>')==1


def test_language_wrappers_follow_selected_languages_and_have_no_output_speaker():
    from tagged_context_reference import request_texts
    assert request_texts('Hello.\n<|speaker:1|>Welcome. Another line.','こんにちは。','en','ja')==(
        '<english>Hello.</english> <|speaker:1|> <english>Welcome. Another line.</english>',
        '<japanese>こんにちは。</japanese>')
    assert request_texts('Hello.','안녕.','unknown','ko')==('Hello.','<korean>안녕.</korean>')


def test_donor_language_tags_follow_audio_parts_even_with_multiline_dialogue():
    from tagged_context_reference import request_texts
    reference,output=request_texts('unused','안녕.','ja','ko',['今。','次。\n続き。','最後。'])
    assert reference=='<japanese>今。</japanese> <|speaker:1|> <japanese>次。\n続き。</japanese> <japanese>最後。</japanese>'
    assert reference.count('<|speaker:1|>')==1
    assert output=='<korean>안녕.</korean>'
