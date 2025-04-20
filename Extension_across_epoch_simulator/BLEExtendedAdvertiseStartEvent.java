
public class BLEExtendedAdvertiseStartEvent extends BLEScheduleEvent{
    private int secondChannel = -1; // the primary or secondary advertising channel
    private int beaconSeq = -1; // the sequence number of the chunk data

    protected BLEExtendedAdvertiseStartEvent(){}
    
    // for BLEnd
    public BLEExtendedAdvertiseStartEvent(int nodeID, double time){
	super(nodeID, time, true);
    }

    public BLEExtendedAdvertiseStartEvent(int nodeID, double time, int secondChannel, int seq){
    super(nodeID, time, true);
    this.secondChannel = secondChannel;
    this.beaconSeq = seq;
    }	    

    public String toString(){
	return (nodeID + " : " + time + " : START ADVERTISE + (" + isActivated + ")");
    }

    // Extended Adv only
    public int getSecondChannel(){
    return secondChannel;
    }
        
    // multi-adv only
    public int getSeq(){
    return beaconSeq;
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
