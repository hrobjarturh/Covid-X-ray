from pgrace import concar
from datetime import datetime
from pgrace.concar.event import Event
import uuid

class EventsManager:
    def __init__(self):
        self.managerID = None
        self.user = None
        self.concar_event = Event('art12dev5')
        
    def prints(self):
        print(self.user)
        print(self.managerID)
        
    def startSession(self):
        self.managerID = uuid.uuid4()
        print(self.managerID)
        
    def setUser(self, user):
        self.user = user
        
    def sendModelUse(self, start_end):
        self.concar_event.SetGovStream(str(self.managerID))
        self.concar_event.SetComment(str('modelUse'))
        self.concar_event.SetCustomStr1(start_end)
        self.concar_event.SendAndClear()
        
    def sendDBUsed(self, nameOfDB, input, match):
        self.concar_event.SetGovStream(str(self.managerID))
        self.concar_event.SetComment(str('dbUsed'))
        self.concar_event.SetCustomStr1(nameOfDB)
        self.concar_event.SetJSONDoc(input)
        self.concar_event.SetCustomNum1(match)
        self.concar_event.SendAndClear()
        
    def sendVerification(self,old_managerID, ver1, ver2):
        # send verification
        self.concar_event.SetGovStream(str(old_managerID))
        self.concar_event.SetComment(str('verification'))
        self.concar_event.SetCustomStr1(ver1)
        self.concar_event.SetCustomStr2(ver2)
        self.concar_event.SendAndClear()
    
        
    
    
    
    