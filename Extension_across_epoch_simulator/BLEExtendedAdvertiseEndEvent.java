public class BLEExtendedAdvertiseEndEvent extends BLEScheduleEvent{
    
    private int secondChannel = -1; // the primary or secondary advertising channel
    private int beaconSeq = -1; // the sequence number of the chunk data

    protected BLEExtendedAdvertiseEndEvent(){}

    // for BLEnd
    public BLEExtendedAdvertiseEndEvent(int nodeID, double time) {
        super(nodeID, time, false);
    }
    
    public BLEExtendedAdvertiseEndEvent(int nodeID, double time, int secondChannel, int beaconSeq){
	super(nodeID, time, false);
    this.secondChannel = secondChannel;
    this.beaconSeq = beaconSeq;
    }

    public String toString(){
	return (nodeID + " : " + time + " : END ADVERTISE + (" + isActivated + ")");
    }

    // Extended Adv only
    public int getSecondChannel(){
    return secondChannel;
    }
    
    // multi-adv only
    public int getSeq(){
    return beaconSeq;
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
