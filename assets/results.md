Based on what I've observed I think having a random goalie is actually better than the trained goalie.  Not in general, just in my current implementation.  Probably has to do with the fact that I'm using the same number of actions for the goalie as the striker so the goalies aren't learning correctly.

This is affecting the strikers as well as they are learning strategies that work against dumb goalies.  For instance, they often boot the ball straight ahead at the start of a game, which works well when the goalies just sit off to the side.

Based on this I'm assigning an advantage rating to the different combinations as follows:
- +1 for [trained striker, trained goalie] vs [random striker, random goalie].  strong, weak   vs weak, strong
- +2 for [trained striker, trained goalie] vs [random striker, trained goalie]. strong, weak   vs weak, weak
- +3 for [trained striker, random goalie]  vs [random striker, trained goalie]. strong, strong vs weak, weak

Here are the results.  The main takeaway is that the team with the training advantage consistently beat the other team.
```
RG: red goalie
BG: blue goalie
RS: red striker
BS: blue striker
T: trained
R: random

match consists of 10 round.  score is given as (red x blue)

episode   RG    BG    RS    BS    score  notes           advantage
100       T     T     T     T     6x1    random none     +0
100       T     R     R     T     1x2    random RS, BG   +3 B
100       R     T     T     R     2x3    random BS, RG   +3 R    * agents are not trained yet here

3100      T     T     T     T     1x6    random none     +0
3100      T     R     R     T     0x6    random RS, BG   +3 B
3100      R     T     T     R     5x1    random BS, RG   +3 R

6700      T     T     T     T     3x6    random none     +0
6700      T     R     R     T     1x5    random RS, BG   +3 B
6700      R     T     T     R     4x1    random BS, RG   +3 R

6800      T     T     T     T     2x4    random none     +0
6800      T     R     R     T     1x6    random RS, BG   +3 B
6800      R     T     T     R     4x1    random BS, RG   +3 R

6900      T     T     T     T     1x7    random none     +0
6900      T     R     R     T     1x7    random RS, BG   +3 B
6900      R     T     T     R     2x0    random BS, RG   +3 R
```


Extended results for one episode
```
6900      T     T     T     T     1x7    random none     +0
6900      R     R     R     R     1x1    random all      +0

6900      R     T     R     T     2x3    random RS, RG   +1 B
6900      T     T     R     T     2x3    random RS       +2 B
6900      T     R     R     T     1x7    random RS, BG   +3 B

6900      T     R     T     R     1x3    random BS, BG   +1 R    * this was not enough of an advantage for red to win
6900      T     T     T     R     2x0    random BS       +2 R
6900      R     T     T     R     2x0    random BS, RG   +3 R
```
