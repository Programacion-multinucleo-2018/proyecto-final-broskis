# Copyright 2018 Arthur Alves Araujo Ferreira and Adrian Biller

# import copy
import numpy as np
import time
import sys
from game2048 import Game2048
from math import sqrt
import random as rnd

VERBOSE = False

class GameState:
    def __init__(self, state=Game2048(_dims=8), move=None, parent=None, implementation='cpu'):
        try:
            self.depth = parent.depth + 1
        except:
            self.depth = 0
        self.state = state
        self.move = move
        self.parent = parent
        self.children = []
        self.ended = False
        if implementation == 'gpu':
            self.implementation = 'gpu'
        else:
            self.implementation = 'cpu'
        return

    def addChild(self, move):
        child = Game2048(np.copy(self.state.board), self.state.score, implementation=self.implementation)
        result = child.action(move)
        if result == 2:
            child.ended = True
        self.children.append(GameState(child, move, self))

    # parametrize this bitch
    def expand(self, amount):
        if self.children == []:
            # Decide which moves are useful
            possibleMoves = [0, 1, 2, 3]
            for play in possibleMoves:
                child = Game2048(np.copy(self.state.board), self.state.score, implementation=self.implementation)
                result = child.action(play)
                if result == 0:
                    # print("Bad action: " + str(play))
                    possibleMoves.remove(play)
                elif result == 2:
                    child.ended = True
                    self.children.append(GameState(child, play, self))
                else:
                    self.children.append(GameState(child, play, self))

            # move to useful places 'amount' times
            move_idx = 0
            while len(self.children) < amount:
                self.addChild(possibleMoves[move_idx])
                move_idx += 1
                if move_idx > len(possibleMoves)-1:
                    move_idx = 0

            # develop all games that were created
            moves = int(sqrt(amount))
            for child in self.children:
                for i in range(moves):
                    result = 0
                    move_tries = 0
                    while result == 0:
                        result = child.state.action(rnd.randint(0,3))
                        move_tries += 1
                        if move_tries > 8:
                            child.ended = True
                            break
        pass

    def act(self):
        ended = True
        for child in self.children:
            if not child.ended:
                ended = False
                break
        if ended:
            self.ended = True
            return False

        endings = {}
        total_count = [None, None, None, None]
        max_score = (None, None)
        for child in self.children:
            # count endings
            if child.ended == True:
                try:
                    endings[child.move] += 1
                except:
                    endings[child.move] = 1
            # average score
            try:
                total_count[child.move] = (total_count[child.move][0] + child.state.score, total_count[child.move][1] + 1)
            except:
                total_count[child.move] = (child.state.score, 1)
            # save move to max score
            if max_score[0] == None or child.state.score > max_score[0]:
                max_score = (child.state.score, child.move)

        avg_score = {}
        max_avg = None
        for i in range(len(total_count)):
            if total_count[i] != None:
                avg_score[i] = float(total_count[i][0] / total_count[i][1])
                try:
                    if avg_score[i] > avg_score[max_avg]:
                        max_avg = i
                except:
                    max_avg = i

        action_votes = [0,0,0,0]
        for key, val in endings.iteritems():
            if (val):
                action_votes[key] -= 1
        action_votes[max_avg] += 2
        action_votes[max_score[1]] += 1

        choice = 0
        for i in range(len(action_votes)):
            if action_votes[i] > action_votes[choice]:
                choice = i

        # print(choice)
        result = self.state.action(choice)
        self.move = choice
        del self.children[:]

    def play(self, child_count):
        total_time = 0
        steps = 0
        while self.ended is not True:
            start = time.time()
            self.expand(child_count)
            self.act()
            end = time.time()
            total_time += (end-start)
            steps += 1
            print(self)
            print("Average time to act: " + str(total_time/steps))
        print("Game finished")
        print("Average time in " + self.implementation + " per action given " + str(child_count) + " games played per action: "+ str(total_time/steps) +" seconds" )

    def __str__(self):
        returnstr = ""
        returnstr += str(self.state.board) + "<- " + "scr: " + str(self.state.score) + " mv: " + str(self.move) + "\n"
        return returnstr

if __name__ == "__main__":
    if (len(sys.argv) == 3):
        if sys.argv[2] == 'cpu':
            searchstate = GameState(implementation='cpu')
        elif sys.argv[2] == 'gpu':
            searchstate = GameState(implementation='gpu')

        searchstate.play(int(sys.argv[1]))
        print(searchstate)
    else:
        print("No arguments or not enough arguments given. Will play using CPU then play using GPU until no more moves can be made.")
        searchstateCPU = GameState(implementation='cpu')
        searchstateCPU.play(100)
        print(searchstateCPU)
        searchstateGPU = GameState(implementation='gpu')
        searchstateGPU.play(100)
        print(searchstateGPU)
