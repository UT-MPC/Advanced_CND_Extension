
// this class is used when we're modeling BLE's three channels...
// we capture the advertsising start event and convert it into three of these, each on a different channel,
// always in order

public class BLEExtendedAdvertiseOneChannelStartEvent extends BLEExtendedAdvertiseStartEvent {

    private int primaryChannel; // the advertising channel on which to advertise
    private int secondChannel = -1;
    private int beaconSeq = -1; // the sequence number of the chunk data

    private BLEExtendedAdvertiseOneChannelStartEvent(){}

    // for BLEnd
    public BLEExtendedAdvertiseOneChannelStartEvent(int nodeID, double time){
        super(nodeID, time);
    }

    public BLEExtendedAdvertiseOneChannelStartEvent(int nodeID, double time, int primaryChannel, int secondChannel, int seq){
        super(nodeID, time, secondChannel, seq);
        this.primaryChannel = primaryChannel;
        this.secondChannel = secondChannel;
        this.beaconSeq = seq;
    }

    public int getPrimaryChannel(){
	return primaryChannel;
    }

    public int getSecondChannel(){
    return secondChannel;
    }

    @Override
    public void process(BLEDiscSimulator simulator){
	simulator.process(this);
    }

    public boolean isBeacon(){
	return true;
    }
}
