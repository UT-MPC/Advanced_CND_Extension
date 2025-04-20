

public class BLEListenEndEvent extends BLEScheduleEvent{

    private BLEListenEndEvent(){}

    public BLEListenEndEvent(int nodeID, double time){
	super(nodeID, time, false);
    }

    public String toString(){
	return (nodeID + " : " + time + " : END LISTEN");
    }

    public void process(BLEDiscSimulator simulator){
	simulator.process(this);
    }

    public boolean isBeacon(){
	return false;
    }
    
}
