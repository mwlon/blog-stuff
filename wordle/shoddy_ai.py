# requirements: python3.8+, numpy
#
# This script is very verbose by default.
# That's to distract you from it being very slow
# for the first iteration.
# But you can make it quiet with the QUIET argument.
#
# Invocation:
# > python3 shoddy_ai.py $WORD_LENGTH $ITERATIONS_TO_SKIP $GOAL [QUIET]
# You may want to skip 1 iteration's computation, because it takes a
# few minutes
# to compute the best first word, and you just want to know the best
# 2nd word given the 1st word you already input.
# For goal, put either MINIMAX or MEAN depending on whether you want
# to guarantee finishing in 6 iterations or finish in (on average) as
# few iterations as possible.
#
# After each iteration, the script prompts you to input what
# word you actually entered and what the result was.
# Say you entered "cats" and it put "a" in yellow and "t" in green.
# Enter this:
# > what was the result? c0a1t2s0
# The number following each letter indicates its color: 0 for gray,
# 1 for yellow, 2 for red.
#
# This script points to the clone's word lists (https://wordlegame.org/).
# If you want the original, hard code USE_ORIGINAL=True

import sys
import numpy as np
from collections import Counter

n = int(sys.argv[1])
skip_iters = int(sys.argv[2])
goal = sys.argv[3].upper()
if goal == 'MINIMAX':
  pass
elif goal == 'MEAN':
  pass
else:
  raise Exception(f'unknown goal {goal}')
quiet = len(sys.argv) >= 5 and sys.argv[4].upper() == 'QUIET'
print(f'running {goal=} for {n=} letters')

bests_len = 5

def filter_to_len(words):
  return [word for word in words if len(word) == n]

USE_ORIGINAL = False
prefix = 'orig_' if USE_ORIGINAL else ''
vocabulary = filter_to_len(open(f'{prefix}vocabulary.txt').read().split('\n'))
answers = filter_to_len(open(f'{prefix}answers.txt').read().split('\n'))

print(f'{len(vocabulary)=} {len(answers)=}')

class Constraints:
  def __init__(self, lbs=Counter(), ubs={}, exact=[None for _ in range(n)], wrong=[set() for _ in range(n)]):
    self.lbs = lbs
    self.ubs = ubs
    self.exact = exact
    self.wrong = wrong

  def with_input_word_result(self):
    while True:
      word = ''
      outcomes = []
      result = input('> what was the result? ').replace(' ', '')
      if len(result) != 2 * n:
        print(f'invalid, must be {2 * n} characters')
        continue

      success = True
      for i in range(n):
        word += result[2 * i]
        
        try:
          outcome = int(result[2 * i + 1])
          if outcome < 0 or outcome > 2:
            raise Exception(f'invalid outcome {outcome}')
        except:
          print('every odd character must be a 0 1 or 2 outcome')
          success = False
          break
        outcomes.append(outcome)

      if success:
        res = self.with_word_result(word, outcomes)
        if res is None:
          print('contradictory input!')
        else:
          return res

  def with_word_result(self, word, outcomes):
    lbs = self.lbs.copy()
    ubs = self.ubs.copy()
    exact = [x for x in self.exact]
    wrong = [x.copy() for x in self.wrong]
    counts = Counter()

    # first apply all the greens because they are preferred
    # e.g. if answer is "BAD" and guess is "DAD", the 2nd "D"
    # will be green and the first will be gray
    for i in range(n):
      l = word[i]
      if outcomes[i] == 2:
        counts[l] += 1
        if l in ubs and counts[l] > ubs[l]:
          return None
        lbs[l] = max(lbs[l], counts[l])
        if exact[i] is not None and exact[i] != l:
          return None
        if l in wrong[i]:
          return None
        exact[i] = l

    # then apply all the grays and yellows
    for i in range(n):
      l = word[i]
      if outcomes[i] == 0:
        if l in lbs and counts[l] < lbs[l]:
          return None
        ubs[l] = counts[l]
      elif outcomes[i] == 1:
        counts[l] += 1
        if l in ubs and counts[l] > ubs[l]:
          return None
        lbs[l] = max(lbs[l], counts[l])
        if exact[i] is not None and exact[i] == l:
          return None
        wrong[i].add(l)

    return Constraints(lbs, ubs, exact, wrong)

  def satisfies(self, word):
    for i in range(n):
      l = word[i]
      exact_l = self.exact[i]
      if exact_l is not None and exact_l != l:
        return False
      if l in self.wrong[i]:
        return False

    counts = Counter()
    for l in word:
      counts[l] += 1
    
    for l, lb in self.lbs.items():
      if counts[l] < lb:
        return False
    for l, ub in self.ubs.items():
      if counts[l] > ub:
        return False

    return True

  def filter_valid(self, words):
    return [word for word in words if self.satisfies(word)]

  def __str__(self):
    return f'exact={self.exact} lbs={self.lbs} ubs={self.ubs}'

def get_outcome_int(guess, answer):
  outcome_int = 0
  skip_inds = set()
  counts = Counter()
  for l in answer:
    counts[l] += 1
  for i in range(n):
    if guess[i] == answer[i]:
      outcome_int += 2 * 3 ** i
      counts[guess[i]] -= 1
      skip_inds.add(i)
  for i in range(n):
    if i in skip_inds:
      continue
    if counts[guess[i]] > 0:
      outcome_int += 3 ** i
      counts[guess[i]] -= 1
  return outcome_int
    
def calc_score(word, valid_answers, constraints):
  score = 0 # higher is better
  accounted = 0
  total_outcomes = 3 ** n
  outcome_counts = np.zeros(total_outcomes, dtype=np.int64)
  for answer in valid_answers:
    outcome_int = get_outcome_int(word, answer)
    outcome_counts[outcome_int] += 1
  
  if goal == 'MINIMAX':
    score = np.max(outcome_counts)
  else:
    # goal is mean
    ps = outcome_counts / len(valid_answers)
    ps = ps[ps > 0]
    score = np.sum(ps * np.log(ps))
  return score
    
constraints = Constraints()
valid_answers = answers
iter_idx = 0
while True:
  valid_answers = constraints.filter_valid(valid_answers)
  valid_answers_set = set(valid_answers)
  print(f'{len(valid_answers_set)} remaining possibilities')
  if len(valid_answers_set) == 1:
    print(f'the word: {valid_answers[0]}')
    break

  if iter_idx >= skip_iters:
    bests = []
    if quiet:
      print('computing best words...')
    for i in range(len(vocabulary)):
      if not quiet:
        print(f'on word {i=} {bests=}')
      word = vocabulary[i]
      # stop calculating if worse than rank (say) 5 minimax
      score = calc_score(word, valid_answers, constraints)
      if score is None:
        continue
      score_tuple = (score, word not in valid_answers_set)
      bests.append((word, score_tuple))
      bests.sort(key=lambda pair: pair[1])
      bests = bests[:bests_len]
    for rank, (word, score_tuple) in enumerate(bests):
      score = score_tuple[0]
      is_possible_answer = not score_tuple[1]
      print(f'\t{rank=} {word=} {score=} {is_possible_answer=}')

  constraints = constraints.with_input_word_result()
  iter_idx += 1
