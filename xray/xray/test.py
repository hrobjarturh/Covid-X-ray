import subprocess
import threading
from io import StringIO 
import select
import sys

class terminal(threading.Thread):
    def run(self):
        self.prompt()

    def prompt(self):
        x = True
        while x:
            select.select((sys.stdin,),(),())
            a = sys.stdin.read(1)
            if not a == '\n':  
                sys.stdout.write(a)
                sys.stdout.flush()
            else:
                x = self.interpret('s')

    def interpret(self,command):
        if command == 'exit':
            return False
        else:
            print('Invalid Command')
        return True

class test(threading.Thread):
    command = 'python3'
    test = StringIO()
    p = subprocess.Popen(command, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    while (p.poll() == None):
        print('test while')
        line = p.stderr.readline()
        print('test while2')
        if not line: break
        print(line.strip())

term = terminal()
testcl = test()
term.start()
testcl.start()