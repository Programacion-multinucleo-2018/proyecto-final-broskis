# Copyright 2018 Arthur Alves Araujo Ferreira and Adrian Biller

import random as rnd
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda
from pycuda.compiler import SourceModule

class Game2048:
    def __init__(self, _board=None, _score=0, _dims=8, implementation='cpu'):
        self.score = np.int32(_score)
        self.dimensions = _dims
        if (_board is not None):
            self.board = _board
        else:
            self.board = np.zeros((self.dimensions, self.dimensions))
            # self.board = np.array([[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]])
            self.putRandom()
        if implementation == 'gpu':
            self.implementation = 'gpu'
        else:
            self.implementation = 'cpu'

    def addScore(self, amount):
        self.score += amount

    def putRandom(self):
        while True:
            pos = (rnd.randint(0,self.dimensions-1), rnd.randint(0,self.dimensions-1))
            if self.board[pos[0]][pos[1]] == 0:
                self.board[pos[0]][pos[1]] = 2 if rnd.uniform(0,1) < 0.9 else 4
                break
            else:
                check_zero = False
                for i in range(self.dimensions):
                    for j in range(self.dimensions):
                        if self.board[i][j] == 0:
                            check_zero = True
                if check_zero == False:
                    return False
        pass
        return True

    # CPU implementation of game actions
    def gravityLine(self, line, direction):
        # line = np.delete(line, np.where(line == 0)[0])
        line_clean = []
        for i in line:
            if (i != 0):
                line_clean.append(i)
        line = np.array(line_clean)
        # check for all rows the blocks under it and if they coincide, add them and pull the rest of the column up
        if (direction == 0):
            for i in range(len(line)-1):
                if (line[i] == line[i+1]):
                    line[i] = line[i]*2
                    self.addScore(line[i])
                    line[i+1] = 0
        elif (direction == 1):
            for i in range(len(line)-1, 0, -1):
                if (line[i] == line[i-1]):
                    line[i] = line[i]*2
                    self.addScore(line[i])
                    line[i-1] = 0

        # new_line = np.delete(line, np.where(line == 0)[0])
        new_line = []
        for i in line:
            if (i != 0):
                new_line.append(i)
        if (direction == 0):
            for i in range(self.dimensions-len(new_line)):
                new_line.append(0)
            # new_line = np.pad(new_line, (0, self.dimensions-len(new_line)), 'constant')
        elif (direction == 1):
            for i in range(self.dimensions-len(new_line)):
                new_line.insert(0,0)
            # new_line = np.pad(new_line, (self.dimensions-len(new_line), 0), 'constant')
        new_line = np.array(new_line)
        return new_line

    def get_func(self):
        mod = SourceModule ("""
            __global__ void gravityLine(int *a, int dims, int *score, int direction)
            {
              __shared__ int *scr;
              int tx = threadIdx.x;
              int ty = threadIdx.y;
              if (ty * dims + tx == 0) {
                scr = &score[0];
              }

              if (blockDim.x == dims) {
                if (direction == 0) {
                  int last_n = -1;
                  for (int i = 0; i < dims; i++) {
                    int idx = i * dims + tx;
                    if (last_n == -1) {
                      if (a[idx] != 0) {
                        last_n = idx;
                      }
                    }
                    else {
                      if (a[idx] != 0) {
                        if (a[idx] == a[last_n]) {
                          a[last_n] = a[idx] * 2;
                          __syncthreads();
                          atomicAdd(scr, a[last_n]);
                          __syncthreads();
                          a[idx] = 0;
                          last_n = -1;
                        }
                        else {
                          last_n = idx;
                        }
                      }
                    }
                  }
                  for (int i = 0; i < dims; i++) {
                    int idx = i * dims + tx;
                    if (a[idx] == 0) {
                      for (int j = i+1; j < dims; j++) {
                        int jdx = j * dims + tx;
                        if (a[jdx] != 0) {
                          a[idx] = a[jdx];
                          a[jdx] = 0;
                          break;
                        }
                      }
                    }
                  }
                } else if (direction == 1) {
                  int last_n = -1;
                  for (int i = dims-1; i >= 0; i--) {
                    int idx = i * dims + tx;
                    if (last_n == -1) {
                      if (a[idx] != 0) {
                        last_n = idx;
                      }
                    }
                    else {
                      if (a[idx] != 0) {
                        if (a[idx] == a[last_n]) {
                          a[idx] = a[idx] * 2;
                          __syncthreads();
                          atomicAdd(scr, a[idx]);
                          __syncthreads();
                          a[last_n] = 0;
                          last_n = -1;
                        }
                        else {
                          last_n = idx;
                        }
                      }
                    }
                  }
                  for (int i = dims-1; i >= 0; i--) {
                    int idx = i * dims + tx;
                    if (a[idx] == 0) {
                      for (int j = i-1; j >= 0; j--) {
                        int jdx = j * dims + tx;
                        if (a[jdx] != 0) {
                          a[idx] = a[jdx];
                          a[jdx] = 0;
                          break;
                        }
                      }
                    }
                  }
                }
              } else {
                if (direction == 0) {
                  int last_n = -1;
                  for (int i = 0; i < dims; i++) {
                    int idx = ty * dims + i;
                    if (last_n == -1) {
                      if (a[idx] != 0) {
                        last_n = idx;
                      }
                    }
                    else {
                      if (a[idx] != 0) {
                        if (a[idx] == a[last_n]) {
                          a[last_n] = a[idx] * 2;
                          __syncthreads();
                          atomicAdd(scr, a[last_n]);
                          __syncthreads();
                          a[idx] = 0;
                          last_n = -1;
                        }
                        else {
                          last_n = idx;
                        }
                      }
                    }
                  }
                  for (int i = 0; i < dims; i++) {
                    int idx = ty * dims + i;
                    if (a[idx] == 0) {
                      for (int j = i+1; j < dims; j++) {
                        int jdx = ty * dims + j;
                        if (a[jdx] != 0) {
                          a[idx] = a[jdx];
                          a[jdx] = 0;
                          break;
                        }
                      }
                    }
                  }
                } else if (direction == 1) {
                  int last_n = -1;
                  for (int i = dims-1; i >= 0; i--) {
                    int idx = ty * dims + i;
                    if (last_n == -1) {
                      if (a[idx] != 0) {
                        last_n = idx;
                      }
                    }
                    else {
                      if (a[idx] != 0) {
                        if (a[idx] == a[last_n]) {
                          a[idx] = a[idx] * 2;
                          __syncthreads();
                          atomicAdd(scr, a[idx]);
                          __syncthreads();
                          a[last_n] = 0;
                          last_n = -1;
                        }
                        else {
                          last_n = idx;
                        }
                      }
                    }
                  }
                  for (int i = dims-1; i >= 0; i--) {
                    int idx = ty * dims + i;
                    if (a[idx] == 0) {
                      for (int j = i-1; j >= 0; j--) {
                        int jdx = ty * dims + j;
                        if (a[jdx] != 0) {
                          a[idx] = a[jdx];
                          a[jdx] = 0;
                          break;
                        }
                      }
                    }
                  }
                }
              }
            }

        """)
        func = mod.get_function("gravityLine")
        return func

    def gravityUp(self):
        if self.implementation == 'gpu':
            func = self.get_func()
            # line = self.board[col, :]
            self.board = self.board.astype(np.int32)
            a_gpu = cuda.mem_alloc(self.board.size * self.board.dtype.itemsize)
            cuda.memcpy_htod(a_gpu, self.board)
            score_arr = np.array([self.score]).astype(np.int32)
            func(a_gpu, np.int32(self.dimensions), cuda.InOut(score_arr), np.int32(0), block=(self.dimensions,1,1))
            # print(score_arr)
            self.score = score_arr[0]
            a_cpu = np.empty_like(self.board)
            cuda.memcpy_dtoh(a_cpu, a_gpu)
            # print(a_cpu)
            self.board = a_cpu
        else:
            for row in range(self.dimensions):
                self.board[:, row] = self.gravityLine(self.board[:, row], 0)

        return

    def gravityRight(self):
        if self.implementation == 'gpu':
            func = self.get_func()
            # line = self.board[col, :]
            self.board = self.board.astype(np.int32)
            a_gpu = cuda.mem_alloc(self.board.size * self.board.dtype.itemsize)
            cuda.memcpy_htod(a_gpu, self.board)
            score_arr = np.array([self.score]).astype(np.int32)
            func(a_gpu, np.int32(self.dimensions), cuda.InOut(score_arr), np.int32(1), block=(1,self.dimensions,1))
            # print(score_arr)
            self.score = score_arr[0]
            a_cpu = np.empty_like(self.board)
            cuda.memcpy_dtoh(a_cpu, a_gpu)
            # print(a_cpu)
            self.board = a_cpu
        else:
            for col in range(self.dimensions):
                self.board[col, :] = self.gravityLine(self.board[col, :], 1)
        # return

    def gravityDown(self):
        if self.implementation == 'gpu':
            func = self.get_func()
            # line = self.board[col, :]
            self.board = self.board.astype(np.int32)
            a_gpu = cuda.mem_alloc(self.board.size * self.board.dtype.itemsize)
            cuda.memcpy_htod(a_gpu, self.board)
            score_arr = np.array([self.score]).astype(np.int32)
            func(a_gpu, np.int32(self.dimensions), cuda.InOut(score_arr), np.int32(1), block=(self.dimensions,1,1))
            # print(score_arr)
            self.score = score_arr[0]
            a_cpu = np.empty_like(self.board)
            cuda.memcpy_dtoh(a_cpu, a_gpu)
            # print(a_cpu)
            self.board = a_cpu
        else:
            for row in range(self.dimensions):
                self.board[:, row] = self.gravityLine(self.board[:, row], 1)
        return

    def gravityLeft(self):
        if self.implementation == 'gpu':
            func = self.get_func()
            # line = self.board[col, :]
            self.board = self.board.astype(np.int32)
            a_gpu = cuda.mem_alloc(self.board.size * self.board.dtype.itemsize)
            cuda.memcpy_htod(a_gpu, self.board)
            score_arr = np.array([self.score]).astype(np.int32)
            func(a_gpu, np.int32(self.dimensions), cuda.InOut(score_arr), np.int32(0), block=(1,self.dimensions,1))
            # print(score_arr)
            self.score = score_arr[0]
            a_cpu = np.empty_like(self.board)
            cuda.memcpy_dtoh(a_cpu, a_gpu)
            # print(a_cpu)
            self.board = a_cpu
        else:
            for col in range(self.dimensions):
                self.board[col, :] = self.gravityLine(self.board[col, :], 0)
        return

    def checkPossibleMoves(self):
        for row in range(self.dimensions):
            for col in range(row%2, self.dimensions, 2):
                # check up
                if col-1 >= 0:
                    if self.board[row][col] == self.board[row][col - 1]:
                        return True
                # check right
                if row+1 < dimensions:
                    if self.board[row][col] == self.board[row + 1][col]:
                        return True
                # check down
                if col+1 < dimensions:
                    if self.board[row][col] == self.board[row][col + 1]:
                        return True
                # check left
                if row-1 >= 0:
                    if self.board[row][col] == self.board[row - 1][col]:
                        return True
        return False

    # Will apply the desired action to board
    # Will return 0 for board unchanged
    # Will return 1 for action applied
    # will return 2  for game ended
    def action(self, direction):
        board_before_action = self.board.tostring()

        if direction == 0:
            self.gravityUp()
        elif direction == 1:
            self.gravityRight()
        elif direction == 2:
            self.gravityDown()
        elif direction == 3:
            self.gravityLeft()

        # check if something changed
        if self.board.tostring() == board_before_action:
            return 0
        else:
            # check if the game is still playable (if it can place random numbers and still has possible moves)
            if not self.putRandom() and not self.checkPossibleMoves():
                return 2
            return 1

def play(verbose = False):
    game = Game2048()
    if verbose: print(game.board, "Score:", game.score)

    while True:
        actionResult = -1
        # Check for input
        ans = str(input("What's your action?"))
        if ans == "":
            continue
        if ans == "0":
            print("You chose exit. Score:", game.score)
            break
        if ans == "8":
            print("You chose Up (^)")
            actionResult = game.action(0)
        elif ans == "6":
            print("You chose Right (>)")
            actionResult = game.action(1)
        elif ans == "5":
            print("You chose Down (v)")
            actionResult = game.action(2)
        elif ans == "4":
            print("You chose Left (<)")
            actionResult = game.action(3)
        else:
            continue

        # exit loop
        if actionResult == 0:
            if verbose: print("Nothing happened.")
        elif actionResult == 2  :
            if verbose:
                if game.score >= 18432:
                    print("You won, but no more possible moves. Score:", game.score)
                else:
                    print("You lost. Score:", game.score)
            break

        if verbose: print(game.board, "Score:", game.score)

if __name__ == "__main__":
    play(True)
