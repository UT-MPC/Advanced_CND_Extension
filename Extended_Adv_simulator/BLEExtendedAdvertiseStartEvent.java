
public class BLEExtendedAdvertiseStartEvent extends BLEScheduleEvent{
    private int secondChannel; // the primary or secondary advertising channel

    protected BLEExtendedAdvertiseStartEvent(){}
    
    public BLEExtendedAdvertiseStartEvent(int nodeID, double time, int secondChannel){
	super(nodeID, time, true);
    this.secondChannel = secondChannel;
    }	    

    public String toString(){
	return (nodeID + " : " + time + " : START ADVERTISE + (" + isActivated + ")");
    }

    public int getSecondChannel(){
    return secondChannel;
    }

    @Override
    public void process(BLEDiscSimulator simulator){
	// because some BLEExtendedAdvertiseStartEvents are inactivated, we have to check to make sure we only
	// process the active ones
	if(isActivated){
	    simulator.process(this);
	}
    }

    public boolean isBeacon(){
	return true;
    }

}
