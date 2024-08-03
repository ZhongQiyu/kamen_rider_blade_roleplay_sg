import pysrt

def extract_lines_from_srt(srt_file):
    subs = pysrt.open(srt_file)
    lines = []
    for sub in subs:
        lines.append(sub.text.strip())
    return lines

lines = extract_lines_from_srt('subtitles.srt')
