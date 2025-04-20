import java.util.Comparator;

public class BLEScheduleEventComparator implements Comparator<BLEScheduleEvent> {

    // returns 0 if the events are the same
    // returns -1 if e1 should come first
    // returns 1 if e2 should come first
    // RULES:
    // all end events should come before start events
    // start listen should come before start advertise
    // end advertise should come before end listen
    @Override
    public  int compare(BLEScheduleEvent e1, BLEScheduleEvent e2){
		int toReturn = Double.compare(e1.getTime(), e2.getTime());
		if(toReturn == 0){
			// the times were the same
			// all end events should come "before" the start events at the same time
			if(!e1.getStart() && e2.getStart()){
			toReturn = -1;
			}
			else if(e1.getStart() && !e2.getStart()){
			toReturn = 1;
			}
			else if(e1.getStart() && e2.getStart()){
			// if they're both start events, then a listen event needs to come before an advertise event
			if((e1 instanceof BLEListenStartEvent) && !(e2 instanceof BLEListenStartEvent)){
				toReturn = -1;
			}
			else if((e2 instanceof BLEListenStartEvent) && !(e1 instanceof BLEListenStartEvent)){
				toReturn = 1;
			}
			}
			else if(!e1.getStart() && !e2.getStart()){
			// if they're both end events, then an advertise event needs to come before a listen event
			if((e1 instanceof BLEExtendedAdvertiseEndEvent) && !(e2 instanceof BLEExtendedAdvertiseEndEvent)){
				toReturn = -1;
			}
			else if((e2 instanceof BLEExtendedAdvertiseEndEvent) && !(e1 instanceof BLEExtendedAdvertiseEndEvent)){
				toReturn = 1;
			}
			}
		}
		return toReturn;
    }
}

    
