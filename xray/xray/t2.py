

'''import subprocess
command = ['./monpoly/monpoly', '-sig','monpoly/art12_app_new/test.sig','-formula','monpoly/art12_app_new/test.mfotl','-negate']
proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

print('o: ',proc.pid,proc.stdout.read().decode("utf-8"))

proc.stdin.write('@1669916093 modelUse (d0f0c693-d623-42a9-a57e-f0874dad1d84,"startDate")'.encode())
print('o1: ',proc.pid,proc.stdout.read().decode("utf-8"))

proc.stdin.write('@1669916094 modelUse (d0f0c693-d623-42a9-a57e-f0874dad1d84,"endDate")'.encode())
print('o2: ',proc.pid,proc.stdout.read().decode("utf-8"))

proc.stdin.write('@1669916458 verification (d0f0c693-d623-42a9-a57e-f0874dad1d84,"ryt","asdfds")'.encode())
print('o3: ',proc.pid,proc.stdout.read().decode("utf-8"))
'''

'''
import subprocess
import pty
import os

cmd = 'python'

master, slave = pty.openpty()
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
stdin_handle = proc.stdin
stdout_handle = proc.stdout



print('checkpoint 1')
stdin_handle.write('2+2'.encode())
stdin_handle.close()
print('checkpoint 2')
print(stdout_handle.readline())'''

#!/usr/bin/env python3
import sys
from subprocess import Popen, PIPE
'''
proc = Popen(['python', '-u'], bufsize=0, shell=True, stdout=PIPE,stdin=PIPE, universal_newlines = True)

#for input_string in ["hello, world!", "and another line", ""]:
print('checkpoint 1')

#print("2+2", file=proc.stdin, flush=True)
proc.stdin.write('2+2')

print('checkpoint 2')
for line in proc.stdout:
    print(line)


print("and another line", file=proc.stdin, flush=True)
print(proc.stdout.readline(), end='')

print('checkpoint 3')
print("", file=proc.stdin, flush=True)
print(proc.stdout.readline(), end='')

'''

''' working
import select

proc = Popen(['python3'],
                        stdin=PIPE,
                        stdout=PIPE,
                        stderr=PIPE,
                        shell=False)

# To avoid deadlocks: careful to: add \n to output, flush output, use
# readline() rather than read()

proc.stdin.write(b'2+2\n')
proc.stdin.flush()
print(proc.stdout.readline().decode())

proc.stdin.write(b'len("foobar")\n')
proc.stdin.flush()
print(proc.stdout.readline().decode())

proc.stdin.close()
proc.terminate()
proc.wait(timeout=0.2)'''

'''
from time import sleep
import sys,os
import subprocess
import select


#  Launches the server with specified parameters, waits however
#  long is specified in saveInterval, then saves the map.


#  Edit the value after "saveInterval =" to desired number of minutes.
#  Default is 30

saveInterval = 1

#  Start the server.  Substitute the launch command with whatever you please.
proc = subprocess.Popen('python3',
                     shell=False,
                     stdin=subprocess.PIPE)

print('checkpoint 1')
#while(True):
select.select((sys.stdin,),(),())

print('checkpoint 2')
#  Comment out these two lines if you want the save to happen silently.
proc.stdin.write(b'2+2\n')
proc.stdin.flush()

print('checkpoint 3')
#  Stop all other saves to prevent corruption.
proc.stdin.write(b'len("foobar")\n')
proc.stdin.flush()
sleep(1)

print('checkpoint 4')
print(proc.stdout.readline().decode())
'''

'''import os
import pty
import select
from subprocess import Popen, STDOUT

master_fd, slave_fd = pty.openpty()  # provide tty to enable
                                     # line-buffering on ruby's side
proc = Popen(['python3'],
             stdout=slave_fd, stderr=STDOUT, close_fds=True)
timeout = .04 # seconds
while 1:
    ready, _, _ = select.select([master_fd], [], [], timeout)
    if ready:
        data = os.read(master_fd, 512)
        if not data:
            break
        print("got " + repr(data))
    elif proc.poll() is not None: # select timeout
        assert not select.select([master_fd], [], [], 0)[0] # detect race condition
        break # proc exited
os.close(slave_fd) # can't do it sooner: it leads to errno.EIO error
os.close(master_fd)
proc.wait()

print("This is reached!")'''


'''proc = Popen('python3', bufsize=0, shell=False, stdout=PIPE,stdin=PIPE, close_fds=False)

for line in proc.stdout:
    print(line)

print("This is most certainly reached!")'''

import subprocess

proc = Popen(['python3','-i'],
                        stdin=PIPE,
                        stdout=PIPE,
                        stderr=PIPE,
                        shell=False)

# To avoid deadlocks: careful to: add \n to output, flush output, use
# readline() rather than read()
proc.stdin.write(b'2+2\n')
proc.stdin.flush()
print(proc.stdout.readline().decode())

proc.stdin.write(b'len("foobar")\n')
proc.stdin.flush()
print(proc.stdout.readline().decode())

proc.stdin.close()
proc.terminate()
proc.wait(timeout=0.2)

