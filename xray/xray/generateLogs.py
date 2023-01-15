from pgrace import concar
from datetime import datetime, timedelta
import logging
from dateutil.parser import parse
from datetime import datetime
import time
import ast
    
class Logger():   
    def __init__(self) -> None:
        self.last_datetime = datetime.now()
    
    def genLogs(self, start=False):
        
        # clear log file
        with open('test.log', 'w'):
            pass
        
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        new_datetime = datetime.now()
        list_of_events = concar.get_events(
            start_date= self.last_datetime,
            end_date=new_datetime,
            name='art12dev35'
        )
        self.last_datetime = new_datetime

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(filename='test.log', level=logging.INFO, format= '%(message)s')
        
            
        for e_index in reversed(range(len(list_of_events))):
            e = list_of_events[e_index]
            e_type = e['comment']
            e_datetime = int(time.mktime(parse(e['createdDate']).timetuple()))
            logging.info(' '.join(['@'+str(e_datetime)]))
            if e_type == 'initialVerification':
                logging.info(' '.join(['@'+str(e_datetime) ,e_type, '('+str(e['govStream'])   +','+'"'+str(e['data']['standardized']['customStr1'])+'"'+')' ]))
                logging.info(' '.join(['@'+str(e_datetime+1)]))
                
            if e_type == 'externalVerification':
                logging.info(' '.join(['@'+str(e_datetime) ,e_type, '('+str(e['govStream'])   +','+'"'+str(e['data']['standardized']['customStr1'])+'"'+')' ]))
                logging.info(' '.join(['@'+str(e_datetime+1)]))
                
            if e_type == 'modelUse':
                logging.info(' '.join(['@'+str(e_datetime) ,e_type, '('+str(e['govStream'])   +','+'"'+str(e['data']['standardized']['customStr1'])+'"'+')' ]))
                if str(e['data']['standardized']['customStr1']) == 'endDate':
                    logging.info(' '.join(['@'+str(e_datetime+1)]))
                
            if e_type == 'dbUsed':
                logging.info(' '.join(['@'+str(e_datetime) ,e_type, '('+str(e['govStream'])   +','+'"'+str(e['data']['standardized']['customStr1'])+'"'+','+str(e['data']['standardized']['customNum1'])+', '+'"'+str(ast.literal_eval(e['data']['standardized']['jsonDoc']))+'"'+')' ]))
                