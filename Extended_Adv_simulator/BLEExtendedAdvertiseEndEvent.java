public class BLEExtendedAdvertiseEndEvent extends BLEScheduleEvent{
    
    private int secondChannel; // the primary or secondary advertising channel

    protected BLEExtendedAdvertiseEndEvent(){}
    
    public BLEExtendedAdvertiseEndEvent(int nodeID, double time, int secondChannel){
	super(nodeID, time, false);
    this.secondChannel = secondChannel;
    }

    public String toString(){
	return (nodeID + " : " + time + " : END ADVERTISE + (" + isActivated + ")");
    }

    public int getSecondChannel(){
    return secondChannel;
    }

    public void process(BLEDiscSimulator simulator){
	// because some BLEExtendedAdvertiseEndEvents are inactivated, we have to check to make sure we only
	// process the active ones

	if(isActivated){
	    simulator.process(this);
	}
    }

    public boolean isBeacon(){
	return true;
    }

}
