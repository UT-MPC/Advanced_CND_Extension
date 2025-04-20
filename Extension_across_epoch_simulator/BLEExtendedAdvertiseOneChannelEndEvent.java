public class BLEExtendedAdvertiseOneChannelEndEvent extends BLEExtendedAdvertiseEndEvent{

    private int primaryChannel; // the advertising channel on which to advertise
    private int secondChannel = -1;
    private int beaconSeq = -1; // the sequence number of the chunk data
    
    private BLEExtendedAdvertiseOneChannelEndEvent(){}
    
    public BLEExtendedAdvertiseOneChannelEndEvent(int nodeID, double time, int primaryChannel){
        super(nodeID, time);
        this.primaryChannel = primaryChannel;
    }

    // for BLEnd
    public BLEExtendedAdvertiseOneChannelEndEvent(int nodeID, double time, int primaryChannel, int secondChannel, int seq){
        super(nodeID, time, secondChannel, seq);
        this.primaryChannel = primaryChannel;
        this.secondChannel = secondChannel;
        this.beaconSeq = seq;
    }

    public String toString(){
	return (nodeID + " : " + time + " : END ADVERTISE ONE CHANNEL + (" + isActivated + ")");
    }

    public int getPrimaryChannel(){
    return primaryChannel;
    }

    // Extended Adv only
    public int getSecondChannel(){
    return secondChannel;
    }
    
    // multi-adv only
    public int getSeq(){
    return beaconSeq;
    }

    public boolean isBeacon(){
	return true;
    }
    
}
