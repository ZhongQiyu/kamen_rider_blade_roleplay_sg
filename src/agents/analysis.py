# coding: utf-8
from __future__ import unicode_literals
from pyknp import Juman

jumanpp = Juman()
result = jumanpp.analysis("下鴨神社の参道は暗かった。")

for mrph in result.mrph_list(): # 各形態素にアクセス
    print("見出し:%s, 読み:%s, 原形:%s, 品詞:%s, 品詞細分類:%s, 活用型:%s, 活用形:%s, 意味情報:%s, 代表表記:%s" \
            % (mrph.midasi, mrph.yomi, mrph.genkei, mrph.hinsi, mrph.bunrui, mrph.katuyou1, mrph.katuyou2, mrph.imis, mrph.repname))# coding: utf-8

