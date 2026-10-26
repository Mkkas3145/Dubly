"""Audible acting cues for faithful dubbing, without scripted dialogue examples."""
PERFORMANCE_INSTRUCTION=(
    '请为TTS完整记录原声的表演细节：在正文相应位置用英文[...]标注听到的'
    '情绪变化、强度、重音、音高和语调、嗓音质感、语速变化、呼吸、停顿、犹豫和尾音。'
    '用标点忠实反映真实语气及反应强度。不要漏掉可听见的表演线索，不添加未听到的内容，不省略原文。'
    '英文线索只能放在正文内，必须保留起始时间戳、说话人编号和结束时间戳。'
)


def transcription_prompt(default):
    return PERFORMANCE_INSTRUCTION+'\n'+default
