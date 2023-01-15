from pgrace import concar
from datetime import datetime
import logging
from dateutil.parser import parse
from datetime import datetime
import time

list_of_events = concar.get_events(
    start_date=datetime.today().date(),
    end_date=datetime.now(),
    name='art12dev5'
)

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename='test.log', level=logging.INFO, format= '%(message)s')
for e_index in reversed(range(len(list_of_events))):
    e = list_of_events[e_index]
    e_type = e['comment']
    e_datetime = int(time.mktime(parse(e['createdDate']).timetuple()))
        
    if e_type == 'verification':
        logging.info(' '.join(['@'+str(e_datetime) ,e_type, '('+str(e['govStream'])  +','+str(e['data']['standardized']['customStr1'])+','+str(e['data']['standardized']['customStr2'])+')' ]))
        
    if e_type == 'modelUse':
        logging.info(' '.join(['@'+str(e_datetime) ,e_type, '('+str(e['govStream'])   +','+str(e['data']['standardized']['customStr1'])+')' ]))
        
    if e_type == 'dbUsed':
        logging.info(' '.join(['@'+str(e_datetime) ,e_type, '('+str(e['govStream'])   +','+str(e['data']['standardized']['customStr1'])+','+str(e['data']['standardized']['customNum1'])+', ('+str(e['data']['standardized']['jsonDoc'])+'))' ]))