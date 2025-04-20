public class BLEExtendedAdvertiseOneChannelEndEvent extends BLEExtendedAdvertiseEndEvent{

    private int primaryChannel; // the advertising channel on which to advertise
    private int secondChannel;
    
    private BLEExtendedAdvertiseOneChannelEndEvent(){}
    
    public BLEExtendedAdvertiseOneChannelEndEvent(int nodeID, double time, int primaryChannel, int secondChannel){
        super(nodeID, time, secondChannel);
        this.primaryChannel = primaryChannel;
        this.secondChannel = secondChannel;
    }

    public String toString(){
	return (nodeID + " : " + time + " : END ADVERTISE ONE CHANNEL + (" + isActivated + ")");
    }

    public int getPrimaryChannel(){
    return primaryChannel;
    }

    public int getSecondChannel(){
    return secondChannel;
    }

    public boolean isBeacon(){
	return true;
    }
    
}
