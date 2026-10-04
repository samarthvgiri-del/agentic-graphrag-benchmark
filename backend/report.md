### Benchmark Comparison Table

| Pipeline | Success Rate | Eval Error Rate | Avg Latency (ms) | Avg Correctness | Avg Completeness | Avg Groundedness | Avg Agent Steps |
|---|---|---|---|---|---|---|---|
| RAG | 19/20 (95.0%) | 20/20 (100.0%) | 21106 | 1.00 | 0.21 | 0.00 | 0.0 |
| GRAPHRAG | 20/20 (100.0%) | 20/20 (100.0%) | 13749 | 0.00 | 0.00 | 0.00 | 0.0 |
| AGENTIC | 20/20 (100.0%) | 20/20 (100.0%) | 37790 | 0.00 | 0.10 | 0.00 | 1.0 |

### Per-Question Analysis

**Q1: According to the provided corpus, how many biathlon events at the 2018 Winter Olympics had more than 73 competitors?**
- **RAG** (36328ms) [success]: The provided evidence includes only one biathlon event at the 2018 Winter Olympics: the Men's individual event (source [3]), which had 86 competitors....
- **GRAPHRAG** (12801ms) [success]: Insufficient evidence to answer the question.
- **AGENTIC** (33301ms) [success]: Insufficient evidence to answer the question.

**Q2: Who won the gold medal in the men's 20 kilometres walk athletics event at the Summer Olympics held immediately before 2016?**
- **RAG** (18003ms) [success]: The provided evidence is insufficient to answer this question.
- **GRAPHRAG** (14364ms) [success]: Insufficient evidence to answer the question.
- **AGENTIC** (35596ms) [success]: The Summer Olympics held immediately before 2016 was the 2012 London Olympics. According to the provided context, the men's 20 kilometres walk at the ...

**Q3: According to the provided corpus, how many shooting events at the 2004 Summer Olympics had more than 37 competitors?**
- **RAG** (15705ms) [success]: 1
- **GRAPHRAG** (10903ms) [success]: Insufficient evidence to answer the question.
- **AGENTIC** (30849ms) [success]: According to the provided context, the shooting events at the 2004 Summer Olympics had the following competitor counts: [chunk_3] shows that only the ...

**Q4: According to the provided corpus, which athletics event at the 2008 Summer Olympics had the highest number of competitors?**
- **RAG** (12802ms) [success]: The provided evidence indicates that the men's 800 metres at the 2008 Summer Olympics had 58 competitors (Source [1]), while the women's 400 metres ha...
- **GRAPHRAG** (9151ms) [success]: Insufficient evidence to answer the question.
- **AGENTIC** (35285ms) [success]: Insufficient evidence to answer the question.

**Q5: Who won the gold medal in the event held at Olympic Weightlifting Gymnasium on 20 September 1988?**
- **RAG** (11529ms) [success]: Naim Süleymanoğlu
- **GRAPHRAG** (15220ms) [success]: Insufficient evidence to answer the question.
- **AGENTIC** (35410ms) [success]: Insufficient evidence to answer the question.

**Q6: Who won the gold medal in the women's 57 kg judo event at the Summer Olympics held immediately before 2020?**
- **RAG** (19595ms) [success]: The provided evidence is insufficient to answer this question. The evidence only includes data for the 2000 and 2008 Summer Olympics (specifically wom...
- **GRAPHRAG** (16790ms) [success]: Insufficient evidence to answer the question.
- **AGENTIC** (46179ms) [success]: Yuka Sato (Japan) [chunk_1]

**Q7: Who won the gold medal in the men's pole vault athletics event at the Summer Olympics held immediately before 2016?**
- **RAG** (10348ms) [success]: Renaud Lavillenie
- **GRAPHRAG** (11910ms) [success]: Insufficient evidence to answer the question.
- **AGENTIC** (33669ms) [success]: Insufficient evidence to answer the question.

**Q8: According to the provided corpus, which sailing event at the 2000 Summer Olympics had the highest number of competitors?**
- **RAG** (16741ms) [success]: The provided evidence indicates that the Open Two Person Keelboat event at the 2000 Summer Olympics had 32 competitors (Source [1]), while the women's...
- **GRAPHRAG** (13497ms) [success]: Insufficient evidence to answer the question.
- **AGENTIC** (35328ms) [success]: Insufficient evidence to answer the question.

**Q9: How many nations competed in Sailing at the 2016 Summer Olympics – Women's RS:X?**
- **RAG** (12674ms) [success]: The provided evidence is insufficient to answer this question.
- **GRAPHRAG** (11339ms) [success]: Insufficient evidence to answer the question.
- **AGENTIC** (34364ms) [success]: Insufficient evidence to answer the question.

**Q10: According to the provided corpus, how many cycling events at the 2000 Summer Olympics had more than 30 competitors?**
- **RAG** (17573ms) [success]: The provided evidence lists two cycling events at the 2000 Summer Olympics: the men's sprint (19 competitors) and the women's 200m Sprint (12 competit...
- **GRAPHRAG** (12389ms) [success]: Insufficient evidence to answer the question.
- **AGENTIC** (31536ms) [success]: Insufficient evidence to answer the question.

**Q11: Who won the gold medal in the event held at Richmond Olympic Oval on 14 February 2010?**
- **RAG** (8639ms) [success]: Martina Sáblíková
- **GRAPHRAG** (11717ms) [success]: Insufficient evidence to answer the question.
- **AGENTIC** (34954ms) [success]: Insufficient evidence to answer the question.

**Q12: According to the provided corpus, how many rowing events at the 2020 Summer Olympics had more than 36 competitors?**
- **RAG** (Nonems) [error]:
- **GRAPHRAG** (12302ms) [success]: Insufficient evidence to answer the question.
- **AGENTIC** (31363ms) [success]: Insufficient evidence to answer the question.

**Q13: Who won the gold medal in the women's 200 metre freestyle swimming event at the Summer Olympics held immediately before 2016?**
- **RAG** (22204ms) [success]: The provided evidence is insufficient to answer this question. The evidence only describes events from the 2016 Summer Olympics (chunks [1], [2], and ...
- **GRAPHRAG** (18117ms) [success]: Insufficient evidence to answer the question.
- **AGENTIC** (44050ms) [success]: Katie Ledecky [chunk_2]

**Q14: Who won the gold medal in the event held at Riocentro – Pavilion 4 on 11–19 August at the 2016 Summer Olympics?**
- **RAG** (21449ms) [success]: The provided evidence is insufficient to answer this question.
- **GRAPHRAG** (14783ms) [success]: Insufficient evidence to answer the question.
- **AGENTIC** (39148ms) [success]: Insufficient evidence to answer the question.

**Q15: Who won the gold medal in the event held at London Velopark on 3 to 4 August at the 2012 Summer Olympics?**
- **RAG** (25996ms) [success]: The provided evidence is insufficient to answer this question.
- **GRAPHRAG** (13717ms) [success]: Insufficient evidence to answer the question.
- **AGENTIC** (48616ms) [success]: Insufficient evidence to answer the question.

**Q16: Who won the gold medal in the men's sprint biathlon event at the Winter Olympics held immediately before 2022?**
- **RAG** (42993ms) [success]: France
- **GRAPHRAG** (14849ms) [success]: Insufficient evidence to answer the question.
- **AGENTIC** (50200ms) [success]: Johannes Høsflot Klæbo [chunk 2]

**Q17: Who won the gold medal in the event held at Royal Artillery Barracks on 28 July 2012?**
- **RAG** (33690ms) [success]: The provided evidence is insufficient to answer this question. The evidence shows an event at the Royal Artillery Barracks on 29 July 2012 (Women's 10...
- **GRAPHRAG** (15898ms) [success]: Insufficient evidence to answer the question.
- **AGENTIC** (30527ms) [success]: Insufficient evidence to answer the question.

**Q18: Who won the gold medal in the men's freestyle 82 kg wrestling event at the Summer Olympics held immediately before 1996?**
- **RAG** (15944ms) [success]: The provided evidence is insufficient to answer this question. The evidence only contains information about the 1996 Summer Olympics wrestling events ...
- **GRAPHRAG** (17364ms) [success]: Insufficient evidence to answer the question.
- **AGENTIC** (52180ms) [success]: Insufficient evidence to answer the question.

**Q19: According to the provided corpus, how many fencing events at the 1988 Summer Olympics had more than 68 competitors?**
- **RAG** (45681ms) [success]: The provided evidence is insufficient to answer this question.
- **GRAPHRAG** (15188ms) [success]: Insufficient evidence to answer the question.
- **AGENTIC** (34571ms) [success]: According to the provided context, fencing at the 1988 Summer Olympics had 3 events: Men's foil (132 competitors), Men's épée (130 competitors), and W...

**Q20: According to the provided corpus, how many alpine skiing events at the 2014 Winter Olympics had more than 63 competitors?**
- **RAG** (13112ms) [success]: The provided evidence lists three alpine skiing events at the 2014 Winter Olympics: Men's combined (50 competitors), Women's combined (36 competitors)...
- **GRAPHRAG** (12681ms) [success]: Insufficient evidence to answer the question.
- **AGENTIC** (38681ms) [success]: Insufficient evidence to answer the question.
