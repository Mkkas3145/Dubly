import numpy as np
from output_sentence_similarity import locate
from tts_output_guard import covered_extras
def words(text):return [{'word':s,'start':i,'end':i+.5} for i,s in enumerate(text.split())]
def matching_encoder(texts):return np.tile(np.array([1.,0.],dtype='float32'),(len(texts),1))
def test_numeric_reading_and_contractions_accept_whole_sentence():
    for expected,heard in [('너 6살이었냐?','너 여섯 살이었냐?'),('할 순 없으니까.','할 수는 없으니까.'),
        ('세로 1번은 항상성.','세로 일번은 항상성.')]:
        w=words(heard)
        found,report=locate(w,{'original_text':'原文','translated_text':expected},covered_extras,matching_encoder)
        assert found==[(0,len(w)-1)] and report['accepted']
def test_semantic_similarity_never_accepts_japanese_replay_just_for_same_meaning():
    found,_=locate(words('私はあなたを信じています。'),{'original_text':'私はあなたを信じています。','translated_text':'나는 당신을 믿고 있어요.'},covered_extras,matching_encoder)
    assert found==[]
def test_low_semantic_score_rejects_similar_spelling():
    def encoder(texts):return np.array([[1.,0.]]+[[0.,1.] for _ in texts[1:]],dtype='float32')
    found,_=locate(words('문을 열지 않았어요.'),{'original_text':'原文','translated_text':'문을 열어 주세요.'},covered_extras,encoder)
    assert not found
def test_semantic_crop_still_requires_proof_for_every_outside_word():
    w=words('나는 당신을 믿고 있어요. 무관한 다른 긴 이야기와 추가된 말입니다.')
    found,_=locate(w,{'original_text':'原文','translated_text':'나는 당신을 믿고 있어요.'},covered_extras,matching_encoder)
    assert not found

def test_semantic_target_with_asr_contraction_can_still_crop_verified_japanese_suffix():
    w=words('난 당신을 믿고 있어요. ミッション')
    segment={'original_text':'原文','translated_text':'나는 당신을 믿고 있어요.',
        '_guard_reference_texts':[{'original_text':'ミッション','translated_text':'미션'}]}
    found,report=locate(w,segment,covered_extras,matching_encoder)
    assert found==[(0,3)] and report['reason']=='semantic_target_with_verified_extras'

def test_words_present_in_requested_sentence_cannot_be_dropped_for_better_embedding():
    w=words('네 감사합니다. 정말 고마워요.')
    segment={'original_text':'原文','translated_text':'네 감사합니다. 정말 고마워요.',
        '_guard_reference_texts':[{'original_text':'네','translated_text':'네'}]}
    found,_=locate(w,segment,covered_extras,matching_encoder)
    assert found==[(0,len(w)-1)]
