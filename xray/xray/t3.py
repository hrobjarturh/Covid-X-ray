import pexpect
import sys
import time


from pgrace import concar
from datetime import datetime, timedelta
import logging
from dateutil.parser import parse
from datetime import datetime
import time
import ast

    
def proc():
    launchcmd = './monpoly/monpoly -sig monpoly/art12_app_new/test.sig -formula monpoly/art12_app_new/test.mfotl -negate -nofilteremptytp -nofilterrel'
    
    all_lines = []
    
    def sendline(monp):
        child.send(monp+'\n')
        child.expect('\n',async_=False)
        #print(child.buffer.decode("utf-8") )
        result = child.before.decode("utf-8")
        print('res: ',result)
        if result:
            all_lines.append(child.before.decode("utf-8").replace('\r', '\n'))
        
    child = pexpect.spawn(launchcmd)
    #child.logfile_read = sys.stdout.buffer
    '''
    while True:
        with open('test.log') as f:
            f = f.readlines()
            
        print('len(f): ',len(f))
        if len(f) == 0:
            print('no new logs')
        else:
            for line in f:
                sendline(line)
                
            # clear log file
            with open('test.log', 'w'):
                pass
            
        print('sleeping ...')
        time.sleep(5)
        '''
        
    sendline('@1670573000;')
    sendline('@1670573586 modelUse (51f92d4e-1241-440d-ac48-696acbe1a653,"startDate")')
    sendline('@1670573999;')
    sendline('@1670573999;')
    sendline('@1670573999;')
    sendline('@1670574999 modelUse (51f92d4e-1241-440d-ac48-696acbe1a652,"startDate")')
    sendline('@1670578999;')
    sendline('@1670578999;')
    sendline('@1670578999;')
    sendline('@1670579990 modelUse (51f92d4e-1241-440d-ac48-696acbe1a651,"startDate")')
    sendline('@1670599990;')
    sendline('@1670599990;')
    sendline('@1670599990;')
    sendline('@1670599990;')
    
    print(all_lines)
    print(' ')
    logs = ""
    violation = ""
    for s in all_lines:
        if 'violation: ' in s:
            violation += s
        else:
            logs += s
        
    print(logs)
    print(' ')
    print(violation)


    child.close()



proc()
#genLogs()
print('end of file...')
