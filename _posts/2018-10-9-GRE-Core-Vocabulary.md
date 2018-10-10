---
layout: post
title: GRE Core-Vocablary
description: "GRE Core Vocablary. Reference(Vocabulary lists): TIANDAO EDU."
modified: 2018-10-9
tags: [Vocabulary]
image:
  feature: abstract-2.jpg
  entry: abstract-2.jpg
  credit: DarGadgetZ
  creditlink: http://www.dargadgetz.com/ios-7-abstract-wallpaper-pack-for-iphone-5-and-ipod-touch-retina/
---

<details open>
<summary style="font-size:25px;">1000 vacabularies, 12 lists</summary>
<div markdown="1">
- [x] [List 1](#1)
- [ ] [List 2](#2)
- [ ] [List 3](#3)
- [ ] [List 4](#4)
- [ ] [List 5](#5)
- [ ] [List 6](#6)
- [ ] [List 7](#7)
- [ ] [List 8](#8)
- [ ] [List 9](#9)
- [ ] [List 10](#10)
- [ ] [List 11](#11)
- [ ] [List 12](#12)
</div>
</details>

<!-- IPA88（采用IPA字符后的标准元音）:
单元音短元音ɪ ə ɒ ʊ ʌ e æ 单元音长元音iː ɜː ɔː uː ɑː
双元音eɪ aɪ ɔɪ aʊ əʊ ɪə eə ʊə 清辅音p t k f θ s ʃ tʃ 浊辅音b d ɡ v ð z ʒ dʒ
其它辅音h m n ŋ l r j w 其它符号ˈˌ[]/ -->

<div class="vocab">
<h2 id="1">List 1</h2>
  <div class="table">
    <div id="petsRow" class="left">
      <!-- PETS LOAD HERE -->
    </div>
    <div id="petsRow" class="right">
      <!-- PETS LOAD HERE -->
    </div>
  </div>
</div>

<div id="petTemplate" style="display: none;">
  <details>
  <summary id="vocabulary">abate /ə'beɪt/</summary>
  <p id="explaination" >*reduce, diminish*</p>
  <p id="instance" align="justify">The rain didn't abate the crowd's enthusiasm for the baseball game.</p>
  </details>
</div>

<script src="https://code.jquery.com/jquery-3.1.1.min.js"></script>
<script type="text/javascript">
  // Load pets.
  $.getJSON('../scripts/vocab.json', function(data) {
    var petTemplate = $('#petTemplate');

    for (i = 0; i < data.length; i ++) {
      if(i%2 == 0){
        var petsRow = $('.left');
      }else{
        var petsRow = $('.right');
      }
      petTemplate.find('#vocabulary').text(data[i].vocabulary);
      petTemplate.find('#explaination').text(data[i].soundmark+" "+data[i].explaination);
      petTemplate.find('#instance').text(data[i].instance);

      petsRow.append(petTemplate.html());
    }
  });
</script>
