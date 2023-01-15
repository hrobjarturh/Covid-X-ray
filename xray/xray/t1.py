import os
import subprocess
import threading
import time


import pty
import sys
from subprocess import Popen, PIPE

#proc = Popen(['./monpoly/monpoly','-sig','monpoly/art12_app_new/test.sig','-formula','monpoly/art12_app_new/test.mfotl','-negate'],stdin=PIPE,stdout=PIPE,stderr=PIPE,shell=False)
proc = Popen(['python3','-i'],stdin=PIPE,stdout=PIPE,stderr=PIPE,shell=False)

# To avoid deadlocks: careful to: add \n to output, flush output, use
# readline() rather than read()

#proc.stdin.write(b'@1670573586 modelUse (51f92d4e-1241-440d-ac48-696acbe1a653,"startDate")\n')
proc.stdin.write(b'2+2\n')

proc.stdin.flush()
print('flush 1')
print(proc.stdout.readline().decode())

#proc.stdin.write(b'@1670573999\n')
proc.stdin.write(b'2+3\n')
proc.stdin.flush()
print('flush 2')

print(proc.stdout.readline().decode())

#proc.stdin.write(b'@1670579999\n')
proc.stdin.write(b'3+3\n')
proc.stdin.flush()
print('flush 3')
  
print(proc.stdout.readline().decode())

proc.stdin.close()
proc.terminate()



#print(os.system("./monpoly/monpoly -sig monpoly/art12_app_new/test.sig -formula monpoly/art12_app_new/test.mfotl -log monpoly/art12_app_new/test.log -negate"))
'''def output_reader(proc):
    print('printing ...',proc.stdout.readline())
    for line in iter(proc.stdout.readline, b''):
      print('line: ', line.rstrip())
      print('got line: {0}'.format(line))'''
#./monpoly/monpoly -sig monpoly/art12_app_new/test.sig -formula monpoly/art12_app_new/test.mfotl -negate

'''master, slave = pty.openpty()
print('Starting process ...')
# process = subprocess.Popen('./monpoly/monpoly -sig monpoly/art12_app_new/test.sig -formula monpoly/art12_app_new/test.mfotl -negate  -nofilteremptytp -nofilterrel', stdin=subprocess.PIPE, stdout=subprocess.PIPE)

#proc = subprocess.Popen(['./monpoly/monpoly','-sig','monpoly/art12_app_new/test.sig','-formula','monpoly/art12_app_new/test.mfotl','-negate','-nofilteremptytp','-nofilterrel'], shell=True, stdin=subprocess.PIPE, stdout=slave)
proc = subprocess.Popen(['python'], shell=True, stdin=subprocess.PIPE, stdout=slave)

print('Process started ...')

stdin_handle = proc.stdin
stdout_handle = os.fdopen(master)

#stdin_handle.write('@1670573586 modelUse (51f92d4e-1241-440d-ac48-696acbe1a653,"startDate")'.encode())
stdin_handle.write('2+2'.encode())
print(stdout_handle.readlines()) #gets the processed input

stdin_handle.close()
proc.wait()
'''
#process.stdin.write('^C'.encode())

'''def send_stdin(log):
  proc.stdin.write(log)
  result = proc.stdout.read()
  print(result)

send_stdin('@1670573586 modelUse (51f92d4e-1241-440d-ac48-696acbe1a653,"startDate")')
send_stdin('@1670573999')
send_stdin('@1670579999')

proc.wait()'''



'''proc.stdin.write('@1670573586 modelUse (51f92d4e-1241-440d-ac48-696acbe1a653,"startDate")')

proc.stdin.write('@1670573999')

proc.stdin.close()
print('result2')
print(proc.stdout.read())
'''



'''
t = threading.Thread(target=output_reader, args=(proc,))
t.start()

proc.stdin.write(b'@1670573586')
proc.stdin.write(b'@1670573584')
proc.stdin.write(b'@1670573582')'''


'''
try:
    time.sleep(0.2)
    for i in range(4):
        time.sleep(0.1)
finally:
    proc.terminate()
    try:
        proc.wait(timeout=0.2)
        print('== subprocess exited with rc =', proc.returncode)
    except subprocess.TimeoutExpired:
        print('subprocess did not terminate in time')
t.join()
'''


'''
proc.stdin.write(b'@1670573586')
proc.stdin.flush()

proc.stdin.write(b'@1670573586 modelUse (51f92d4e-1241-440d-ac48-696acbe1a653,"startDate")')



process.stdin.write('@1670573587'.encode())
process.stdin.write('@1670573586 modelUse (51f92d4e-1241-440d-ac48-696acbe1a653,"startDate")'.encode())

process.stdin.write('@1670573591'.encode())

process.stdin.write('@1670573599'.encode())

process.stdin.write('@1670573999'.encode())
process.stdin.write('@1670573999'.encode())

print('while True: ...')
while True:
  line = process.stdout.readline()
  if not line:
    break
  #the real code does filtering here
  print("test:", line.rstrip())
  '''

print('End of file ...')




