

public class BLEListenStartEvent extends BLEScheduleEvent{

    
    private int channel; // the advertising channel on which to listen
    
    private BLEListenStartEvent(){}

    public BLEListenStartEvent(int nodeID, double time, int channel){
	super(nodeID, time, true);
	this.channel = channel;
    }

    public int getChannel(){
	return channel;
    }

    public String toString(){
	return (nodeID + " : " + time + " : START LISTEN : " + channel);
    }

    public void process(BLEDiscSimulator simulator){
	simulator.process(this);
    }

    public boolean isBeacon(){
	return false;
    }
}
