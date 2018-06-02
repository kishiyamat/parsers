# 句構造解析

心理言語学の論文を読んでいて詰むポイントは
定量的に反応を予測できなかったり、
理論が視覚的に理解できなかったりする点です。
実際に計算機に解析器を実装しようとなると、
その過程では
言語模型に厳密な定義を形式的に与えねばなりません。
言語とは何か、予測とは何か、処理負荷とは何か、記憶とは何か。
そういった変数に中身を与えて模型を作ります。
そして現実の解析器に刺激を与え、
模型が現実に倣えるかを検証していきたいのです。

先行研究でもATNやEarley法の名前はしばしば見ます。
再分析の話では順位再付与が関連しますし、
これまでの心理言語学で
積極的に計算機を使おうという取り組みは行われています。
また、モデルを組んで手を動かして実装してみる過程は
理解を助けますし、自分に関して言えば楽しい作業でもあります。
そしてなにより、人間の頭の中には実装済みの解析器があるので、
実験によって模型が正しく動くかを確かめられます。
いわば逆行工学のようなものだと自分は認識しています。

というわけで、丁寧に@okumura2017を読み進めて行きつつ、
心理言語学との関連も見出していきます。
まずは@hale2001があるのですが、
その中には文脈自由文法を派生したEarly法がでてきます。
そこまでの実装をするために、
まずは肩慣らしとしてCKYアルゴリズムを実装し、
Earley法へと突撃します。
@hale2001の時代のEarly法よりも
@jurafsky2014の時代の方が進んでいるので、
そのアルゴリズムをどう適用するかも焦点に入れます。

## 句構造

構文構造の表現法の一つである。
文が名詞句と動詞句から構成され、
名詞句は冠詞と名詞から構成され、
というあれである。
"Time flies like an arrow"
には２つの構造が与えられるが、
どのようにして構造を決定するかが問題となる。
心理言語学的には構造的曖昧性である。

## 文脈自由文法

いわゆるType2 Grammar。
非終端記号の集合（SとかNPとかVとか）、
終端器号の集合（犬とか走るとか）、
一つの非終端記号から終端器号と非終端記号の集合への書き換え規則
で構成する。
規則は構文規則と語彙規則に分け、
構文規則は非終端から非終端、
語彙規則は非終端から終端への遷移を定義する。
構文規則を再帰的に適用して無限の文字列を生成できる。
与えられた単語列に対してどう
規則を適用して句構造えるのか、
が解析器の役目となる。
以下は文脈自由文法を基礎に持つ
解析器としてCKY法とEarley法を説明し実装する。
対象言語は英語からはじめ、
追って日本語へ応用する。
実装言語はPython3を検討している。

### CKY法

つくってあそぼ、的な何か。

視覚的に理解できる。アルゴリズムは目で見るに限る。
長尾真の自然言語処理もわかりやすそう。

[Grammar (needs to be in CNF) ](http://lxmls.it.pt/2015/cky.html)
[The CYK Algorithm](https://www.xarg.org/tools/cyk-algorithm/)
[Chart Parsing: The CYK Algorithm](http://staff.icar.cnr.it/ruffolo/progetti/projects/09.Parsing%20CYK/presentazione%20imp--2007_inf2a_L17_slides.pdf)

まずは以下の入力を構文解析すると仮定する。

```python
# ちゃんと仮想環境 'parser' を有効にしましょう。
user_input = "I saw a girl with a telescope"
user_input_list = user_input.split()
print(user_input_list)
len(user_input_list)
```

非終端記号、前終端記号、終端器号の集合を句構造規則に組込む。
文脈自由文法には辞書規則の集合と句構造規則の集合を持つ。
辞書規則の集合は前終端記号から終端記号への関数とする。
句構造規則を非終端記号から非終端、前終端記号の集合への関数とする。
実際に使用する際は２つの非終端記号から親を参照できる形が嬉しい。

```python
cfg_phrase = (
    # 句構造規則
    # こちらは順序対
    ('S' ,( ('NP','VP'),)),
    ('NP',  ('DET','N'),
            ('NP','PP'),),
    ('VP',( ('V','NP') ,
            ('V','NP'),)),
    ('PP',( ('PREP','NP'),)),
    )

cfg_lexicon = (
    # 辞書規則
    # 単語の集合への関数
    ('N'   ,('I','girl','telescope',)),
    ('NP'  ,('I','girl','telescope',)),
    ('V'   ,('saw',)),
    ('VP'  ,('saw',)),
    ('DET' ,('a',)),
    ('PREP',('with',)),
    )
```

```python
# まずは２つの構造の非終端記号から１つの非終端記号を返す関数
def merge(two_element):
    pre = two_element[0]
    post= two_element[1]
    parent=[(cfg[0]) for cfg in cfg_phrase if two_element in cfg[1]]
    # cfgに則っている前提
    if len(parent) == 0:
        return([])
    else:
        return(parent)

def tag(lexicon):
    parent=[(cfg[0]) for cfg in cfg_lexicon if lexicon in cfg[1]]
    # cfgに則っている前提
    if len(parent) == 0:
        return([])
    else:
        return(parent)

merge(('PREP','NP'))
tag('saw')
```

長尾真の『自然言語処理』は三角行列を使っている。
行列はnumpyを使ったほうが書くのも読むのも見るのも楽。
まずは三角行列をnumpyで生成します。
文字列は "I saw a girl with a telescope"
とします。
文字列の長さは`len()`で出せるので。


```python
import numpy as np
import itertools

cfg_phrase = (
    # 句構造規則
    # こちらは順序対
    ('S' ,( ('NP','VP'),)),
    ('NP',( ('DET','N'),
            ('NP','PP'),)),
    ('VP',( ('V','NP') ,
            ('V','NP'),)),
    ('PP',( ('PREP','NP'),)),
    )

cfg_lexicon = (
    # 辞書規則
    # 単語の集合への関数
    ('N'   ,('I','girl','telescope',)),
    ('NP'  ,('I','girl','telescope',)),
    ('V'   ,('saw',)),
    ('VP'  ,('saw',)),
    ('DET' ,('a',)),
    ('PREP',('with',)),
    )

def merge(two_element):
    pre = two_element[0]
    post= two_element[1]
    parent=[(cfg[0]) for cfg in cfg_phrase if two_element in cfg[1]]
    # cfgに則っている前提
    if len(parent) == 0:
        return("")
    else:
        return(parent[0])

def tag(lexicon):
    parent=[(cfg[0]) for cfg in cfg_lexicon if lexicon in cfg[1]]
    # cfgに則っている前提
    if len(parent) == 0:
        return([])
    else:
        return(parent)

user_input = "I saw a girl with a telescope"
user_input_list = user_input.split()
print(user_input_list)
l = len(user_input_list)

# 要はiとかjとかでアクセスできれば良いわけで。
cky_triangle= [[[i,j] for j in range(l)] for i in range(l)]
print(cky_triangle)
for i in cky_triangle:
    print(i)

# 組み合わせを作る
for i, word in enumerate(user_input_list):
    cky_triangle[i][i]=tag(word)

for i in cky_triangle:
    print(i)

# これはあってる。最初から
for d in range(0,l-2):
    for i in range(0,l-d):
        print('i=',i)
        j = i + d + 1
        print('j=',j)
        for k in (i,j):
            print('k=',k)
            # A->BC in P
            ik=cky_triangle[i][k]
            print(ik)
            i1j=cky_triangle[i+1][j]
            print(i1j)
            p = list(itertools.product(ik,i1j))
            print("p=")
            print(p)
            cky_triangle[i][j] = list(map(merge,p))

cky_triangle[1,1]=2

```

* [PCFG構文解析法](https://www.slideshare.net/YusukeOda1/pcfg-51424675)
* [【python】CKY法をpythonで実装](https://hayataka2049.hatenablog.jp/entry/2018/02/19/044452)

### Earley法

## 確率文脈自由文法
## 識別モデルによる順位再付与
## 遷移型モデルによる句構造解析
## 評価法
## 近年の動向

## The Cocke-Kasami-Younger (CKY) algorithm 

[CKY Parser for Japanese](http://www2.hawaii.edu/~chin/661F12/Projects/rbungard.pdf)
