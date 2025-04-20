
// this class is used when we're modeling BLE's three channels...
// we capture the advertsising start event and convert it into three of these, each on a different channel,
// always in order

public class BLEExtendedAdvertiseOneChannelStartEvent extends BLEExtendedAdvertiseStartEvent {

    private int primaryChannel; // the advertising channel on which to advertise
    private int secondChannel;

    private BLEExtendedAdvertiseOneChannelStartEvent(){}

    public BLEExtendedAdvertiseOneChannelStartEvent(int nodeID, double time, int primaryChannel, int secondChannel){
	super(nodeID, time, secondChannel);
	this.primaryChannel = primaryChannel;
    this.secondChannel = secondChannel;
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
