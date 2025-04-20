import java.util.PriorityQueue;
import java.util.ArrayList;
import java.util.Vector;
import java.util.HashMap;
import java.util.Iterator;
import java.io.IOException;
import java.io.OutputStream;
import java.io.ObjectOutput;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.ObjectOutputStream;
import java.io.InputStream;
import java.io.ObjectInput;
import java.io.FileInputStream;
import java.io.BufferedInputStream;
import java.io.ObjectInputStream;
import java.util.HashSet;
import java.util.Set;
import java.util.stream.Stream;
import java.util.Map;
import java.util.HashMap;

public class BLEDiscSimulator{

    private ArrayList<BLESchedule> nodeSchedules = new ArrayList<BLESchedule>();

    // listeners do discovery
    private ArrayList<ListenEventRecord> currentListeners = new ArrayList<ListenEventRecord>();
    // advertisers are discovered
    //private ArrayList<AdvertisingEventRecord> currentAdvertisers = new ArrayList<AdvertisingEventRecord>();
    private ArrayList<ExtendedAdvertisingEventRecord> currentExtendedAdvertisers = new ArrayList<ExtendedAdvertisingEventRecord>();

    private ArrayList<CompletedDiscovery> successfulDiscoveries = new ArrayList<CompletedDiscovery>();
    private ArrayList<Collision> collisions = new ArrayList<Collision>();

    private BLEDiscSimulatorOptions options;
    private BLEDiscLogger logger;
	private BLEDiscLogger dc;

    private PriorityQueue<BLEScheduleEvent> eventQueue =
	new PriorityQueue<BLEScheduleEvent>(20000, new BLEScheduleEventComparator()); // increase for chunks

    private int numNodes;
    private double simulationTime;

    private int logStyle;
    
    //simulation time is specified in number of epochs
    public BLEDiscSimulator(String propertiesFile, String logfile, String dcfile){
		this.options = new BLEDiscSimulatorOptions(propertiesFile);
		if (this.options == null) {
			throw new IllegalStateException("Failed to load simulation options.");
		}
		this.logger = new BLEDiscLogger(logfile);
		this.dc = new BLEDiscLogger(dcfile);
		this.logStyle = options.getLogStyle();
		if(logStyle == BLEDiscSimulatorOptions.LOG_STYLE_VERBOSE){
			logger.log(options.toString());
		}
		this.numNodes = options.getNumNodes();
		this.simulationTime = options.getSimulationTime();

		if(options.loadSchedulesFromFile()){
			loadSchedules(options.getScheduleLoadFile());
		}
		else{
			createSchedules();
		}
		
		if(options.saveSchedulesToFile()){
			writeSchedules(options.getScheduleSaveFile());
		}

		/*for(BLESchedule schedule : nodeSchedules){
			schedule.printSchedule();
			}*/
		
		/*if(options.showSchedules()){
			// TODO: how much should we display, and how is it dependent on the protocol?
			ShowSchedules display = new ShowSchedules("BLEnd Node Schedules", nodeSchedules, 4000);
			display.pack();
			RefineryUtilities.centerFrameOnScreen(display);
			display.setVisible(true);
			}*/
    }

    private void createSchedules(){
		// first, I need to create the schedules for each node in the simulation

		for(int idCounter = 0; idCounter<numNodes; idCounter++){
			BLESchedule schedule = null;
			if (options.getProtocol() == BLEDiscSimulatorOptions.PROTOCOL_BLEND) {
				schedule = new BLEndAESchedule(idCounter, options, simulationTime, null);
			} else if(options.getProtocol()==BLEDiscSimulatorOptions.PROTOCOL_NIHAO){
				schedule = new NihaoAESchedule(idCounter, options, simulationTime, null);
			}  else if(options.getProtocol()==BLEDiscSimulatorOptions.METHOD_EXTENDED_NIHAO){
				schedule = new NihaoExtendedAESchedule(idCounter, options, simulationTime, null);
			} else {
				// ExtendedAE or Extended
				schedule = new ExtendedAESchedule(idCounter, options, simulationTime, null);
			}
			if(schedule != null){
				nodeSchedules.add(schedule);
			}
		}

		makeEventQueue();
    }

    private void makeEventQueue(){
		// then I need to add each event in each schedule to the event queue
		// EVENTS DELETED events = new ArrayList<BLEScheduleEvent>();
		for(BLESchedule nodeSchedule : nodeSchedules){
			ArrayList<BLEScheduleEvent> toAdd = nodeSchedule.getSchedule();
			// EVENTS DELETED toAdd.forEach(event -> events.add(event));
			toAdd.forEach(event -> eventQueue.add(event));
		}
		// then, I want them in order, so sort them.
		// EVENTS DELETED (but it's ok... it's a priority queue now) events.sort((e1, e2) -> BLEScheduleEvent.compareEvents(e1,e2));
    }

    private void simulate(){
		// events contains the sequential list of events to simulate
		// I'm not using an iterator because eventually I want to be able to insert into this list
		// while I traverse it.
		// EVENTS DELETED events.forEach(event -> processEvent(event));
		BLEScheduleEvent nextEvent = eventQueue.poll();
		while(nextEvent != null){
			processEvent(nextEvent);
			nextEvent = eventQueue.poll();
		}

		// we can only safely log discoveries once the simulation has completed (because of the funky
		// way I'm "undoing" discoveries in the case of collisions.
		
		// because of the way the list is implemented, the discovery events should be in order by time
		// however, we all know what happens when you assume...
		// so let's sort them, just to be sure.
		successfulDiscoveries.sort((d1, d2) -> Double.compare(d1.timestamp, d2.timestamp));

		computeCDFData();
		if(logStyle == BLEDiscSimulatorOptions.LOG_STYLE_BRIEF){
			logger.log("A : B : time\n");
		}
		Vector<String> discoveryStrings = new Vector<String>();
		for(CompletedDiscovery cd : successfulDiscoveries){
			if(logStyle == BLEDiscSimulatorOptions.LOG_STYLE_BRIEF){
			logger.log(cd.discovererID + " : " + cd.discoveredID + " : " + cd.timestamp + "\n");
			}
			else if(logStyle == BLEDiscSimulatorOptions.LOG_STYLE_VERBOSE){
			logger.log("N: " + cd.timestamp + " : " + cd.discovererID + " : " + cd.discoveredID + "\n");
			}	    
		}
			logger.close();
		if(options.printStatistics()){
			nodeSchedules.forEach(nodeSchedule -> printStatistics(nodeSchedule));
		}
		dc.close();
    }

    private void printStatistics(BLESchedule nodeSchedule){
		int nodeID = nodeSchedule.getNodeID();
		String stats = nodeSchedule.getDutyCycle() + ","
		+ nodeSchedule.getConsumption() + ","
		+ nodeSchedule.getConsumptionWithIdleCost(0.001) + ","
		+ nodeSchedule.getConsumptionWithIdleCost(0.339)+"\n";
		//+ nodeSchedule.getConsumptionWithIdleCost(0.080)+"\n";

		// System.out.println(stats);
		dc.log(stats);

		/*System.out.println("Duty cycle of node " + nodeID + ": " +
				nodeSchedule.getDutyCycle());
		System.out.println("Node " + nodeID + "'s discovery rate: " +
				computeDiscoveryRate(nodeID, simulationTime/2)); 
		System.out.println("Node " + nodeID + "'s average discovery latency: " +
		computeAverageDiscoveryLatency(nodeID, simulationTime/2));*/
    }

    // if we're just printing the data for the CDF generation in R, we just need to, for each node,
    // find the *first* time it discovered each other time (starting at some randomly selected "contact"
    // time. We randomly select the contact time as being between T (or t) (for warmup) and simulationTime/2
    private void computeCDFData(){
		double intervalSize = (simulationTime/2) - 2*options.getW();
		double contactTime = (Math.random() * intervalSize) + 2*options.getW();
		//System.out.println("SETTING CONTACT TIME EXPLICITLY!");
		//int contactTime = 8638;
		// buono. the discoveries are in order, so we scroll through them to find the first event that happens
		// at or after the contact time
		Iterator<CompletedDiscovery> it = successfulDiscoveries.iterator();
		CompletedDiscovery next = null;
		if(it.hasNext()){
			next = it.next();
		}
		while(next != null && next.timestamp < contactTime){
			if(it.hasNext()){
			next = it.next();
			}
			else{
			next = null;
			}
		}
		// if(next != null){
		double[][] discoveryLatencies = new double[numNodes][numNodes];
		for(int i = 0; i<numNodes; i++){
		for(int j = 0; j<numNodes; j++){
			discoveryLatencies[i][j] = Double.MAX_VALUE;
		}
		}
		// this will count how many things are in the 2D array. When this counter gets to
		// n*n, we're done (TODO: not true for unidirectional discovery...)
		int count = 0;
		for(int i = 0; i<numNodes; i++){
		// a node always discovers itself in 0 time
		discoveryLatencies[i][i] = 0;
		count++;
		}
		if(next != null){
			Map<String, Set<Integer>> partialDiscoveries = new HashMap<>(); // multi-adv check
			// Map<String, Integer> partialDiscoveries = new HashMap<>();

			// next references the first discovery event after the selected contactTime
			// now we'll just scroll through the discovery events, using the info to fill up our 2D array
			// we can quit when either (a) the array is full (i.e., count = n*n) or we run out of events
			while (next != null && (count < (numNodes * numNodes))) {
				int discovererID = next.discovererID;
				int discoveredID = next.discoveredID;

				double discoveryTime = next.timestamp - contactTime;
				String key = discovererID + "_" + discoveredID;

				int chunkNumber = next.seq;
				Set<Integer> curPartialDiscovery = partialDiscoveries.get(key);
				if (curPartialDiscovery == null) {
					// curPartialDiscovery = new Set<Integer>(discovererID, discoveredID);
					curPartialDiscovery = new HashSet<>();
				}
				curPartialDiscovery.add(chunkNumber);
				partialDiscoveries.put(key, curPartialDiscovery); 
				// Check if all the chunks of discoveredNode is discovered and then add to successfulDiscoveries if so
				int n;
				if ((options.getChunks() * 0.8) % 1 > 0) {
					n = (int) Math.floor(options.getChunks() * 0.8) + 1;
				} else {
					n = (int) Math.floor(options.getChunks() * 0.8);
				}
				if (curPartialDiscovery.size() >= n) {
				// if (curPartialDiscovery >= n) {
					if (discoveryLatencies[discovererID][discoveredID] > discoveryTime) {
						discoveryLatencies[discovererID][discoveredID] = discoveryTime;
						count++;
					}
				}

				if (it.hasNext()) {
					next = it.next();
				} else {
					next = null;
				}
			}
			/*if(count < (numNodes * numNodes)){
			System.out.println("OOPS!");
			}*/
		}
			
		// if we run out of events without getting to n*n,
		// anything that is still MAX_INT indicates a discovery that never happened
		if(logStyle == BLEDiscSimulatorOptions.LOG_STYLE_CDF){
		for(int i = 0; i<numNodes; i++){
			for(int j = 0; j<numNodes; j++){
			if(i != j){
				// logger.log(discoveryLatencies[i][j] + "," + i + "," + j + "\n");
				logger.log(discoveryLatencies[i][j] + "\n");
			}
			}
		}
		}
		// logger.log("----------\n");
		// }
    }

    // this is providing me some polymorphic static binding. It's ok. I read about it on the Internet.
    private void processEvent(BLEScheduleEvent bse){
		// this calls to the base class, but the method is abstract, implemented in only the derived classes
		// each derived class calls back to the below "process" method with the correct derived type
		// is this easier than using instanceof? Meh.
		if(!bse.isInWPScan() && !bse.isInWPAdv() && !bse.isPkLoss()) {
			bse.process(this);
		}
    }

    public void process(BLEListenStartEvent blse){
		// create a record for the listener in case he discovers someone
		ListenEventRecord myListenEventRecord = new ListenEventRecord(blse.getNodeID(), blse.getChannel(), blse.getTime());
		// only add to currentListeners when it is not in Warmup interval
		currentListeners.add(myListenEventRecord);
		if(logStyle == BLEDiscSimulatorOptions.LOG_STYLE_VERBOSE){
			logger.log("L" + blse.getChannel() + ": " + blse.getTime() + " : " + blse.getNodeID() + "\n");
		}
    }

    public void process(BLEListenEndEvent blee){
		// only process this when it is not in Warmup interval
		// remove the record for the listener
		ListenEventRecord myListenEventRecord = getListenEventRecordForNodeID(blee.getNodeID());
		currentListeners.remove(myListenEventRecord);

		// if myDiscoveries.discoveryEvents.size() != 0, I successfully discovered someone!
		for(ListenEventRecord.DiscoveryEvent de : myListenEventRecord.discoveryEvents){
			successfulDiscoveries.add(new CompletedDiscovery(blee.getNodeID(), de.discoveredNode, de.timestamp, de.seq));
		}
    }

	// HC: CHANGING!
    public void process(BLEExtendedAdvertiseStartEvent base){
		// if we're not modeling the three channels, we do what we originally did, which is to
		// just assume that the beacon lasts for entire time (i.e., 3ms), that any listener listening on any channel
		// will hear it, as long as they are listening for the whole beacon time
		double time = base.getTime();

		if (logStyle == BLEDiscSimulatorOptions.LOG_STYLE_VERBOSE) {
			logger.log("A: " + base.getTime() + " : " + base.getNodeID() + "\n");
		}
		if (!options.modelChannels()) {
			boolean isExtended = options.getProtocol() == BLEDiscSimulatorOptions.METHOD_MIX || options.getProtocol() == BLEDiscSimulatorOptions.METHOD_EXTENDED || options.getProtocol() == BLEDiscSimulatorOptions.METHOD_EXTENDED_NIHAO;
			processSingleBeacon(base, 0, base.getSecondChannel(), isExtended); // just use channel 0 for primary channel;
		} else {
			// we need to create three beacons on the three different beacon channels.
			// then we need to insert the second two into the event queue and immediately process the first
			// TODO: instead of doing +1, +2, and +3, we should really do b, 2b, and 3b
			BLEExtendedAdvertiseOneChannelStartEvent beacon1 = new BLEExtendedAdvertiseOneChannelStartEvent(base.getNodeID(), base.getTime(),
			BLEScheduleEvent.ADVERTISEMENT_CHANNEL_ONE, base.getSecondChannel(), base.getSeq());
			BLEExtendedAdvertiseOneChannelEndEvent endbeacon1 = new BLEExtendedAdvertiseOneChannelEndEvent(base.getNodeID(), base.getTime() + 1,
			BLEScheduleEvent.ADVERTISEMENT_CHANNEL_ONE, base.getSecondChannel(), base.getSeq());
			BLEExtendedAdvertiseOneChannelStartEvent beacon2 = new BLEExtendedAdvertiseOneChannelStartEvent(base.getNodeID(), base.getTime() + 1,
			BLEScheduleEvent.ADVERTISEMENT_CHANNEL_TWO, base.getSecondChannel(), base.getSeq());
			BLEExtendedAdvertiseOneChannelEndEvent endbeacon2 = new BLEExtendedAdvertiseOneChannelEndEvent(base.getNodeID(), base.getTime() + 2,
			BLEScheduleEvent.ADVERTISEMENT_CHANNEL_TWO, base.getSecondChannel(), base.getSeq());
			BLEExtendedAdvertiseOneChannelStartEvent beacon3 = new BLEExtendedAdvertiseOneChannelStartEvent(base.getNodeID(), base.getTime() + 2,
			BLEScheduleEvent.ADVERTISEMENT_CHANNEL_THREE, base.getSecondChannel(), base.getSeq());
			BLEExtendedAdvertiseOneChannelEndEvent endbeacon3 = new BLEExtendedAdvertiseOneChannelEndEvent(base.getNodeID(), base.getTime() + 3,
			BLEScheduleEvent.ADVERTISEMENT_CHANNEL_THREE, base.getSecondChannel(), base.getSeq());

			// all I need to do with the priority queue is add these new events!
			eventQueue.add(beacon1);
			eventQueue.add(endbeacon1);
			eventQueue.add(beacon2);
			eventQueue.add(endbeacon2);
			eventQueue.add(beacon3);
			eventQueue.add(endbeacon3);
		}
    }

    public void process(BLEExtendedAdvertiseOneChannelStartEvent baocse){
		if (logStyle == BLEDiscSimulatorOptions.LOG_STYLE_VERBOSE) {
			logger.log("A" + baocse.getPrimaryChannel() + " " + baocse.getSecondChannel() + ": " + baocse.getTime() + " : " + baocse.getNodeID() + "\n");
		}
		if (options.getProtocol() == BLEDiscSimulatorOptions.METHOD_MIX || options.getProtocol() == BLEDiscSimulatorOptions.METHOD_EXTENDED || options.getProtocol() == BLEDiscSimulatorOptions.METHOD_EXTENDED_NIHAO) {
			processSingleBeacon(baocse, baocse.getPrimaryChannel(), baocse.getSecondChannel(), true); //extended + AE
		} else {
			processSingleBeacon(baocse, baocse.getPrimaryChannel(), baocse.getSecondChannel(), false);
		}
		// processSingleBeacon(baocse, baocse.getPrimaryChannel(), baocse.getSecondChannel());
    }

	private void processSingleBeacon(BLEExtendedAdvertiseStartEvent base, int primaryChannel, int secondChannel, boolean isExtended) {
		// create a record to store the other devices that have discovered me
		// this is actually so we can correct for conflicts
		ExtendedAdvertisingEventRecord myAdvertisingEventRecord = new ExtendedAdvertisingEventRecord(base.getNodeID(), primaryChannel, secondChannel, base.getTime());
	
		// when I start advertising, I assume that every listener discovers me. If they're not still
		// listening when I stop, I'll remove them. Also, if we detect a collision, we'll remove them
	
		// cycle through all of the current advertisers
		for (ExtendedAdvertisingEventRecord otherAdvertiser : currentExtendedAdvertisers) {
	
			boolean primaryOverlap, secondaryOverlap;
	
			if (isExtended) {
				primaryOverlap = otherAdvertiser.startTime + options.getB() >= base.getTime() && otherAdvertiser.primaryChannel == primaryChannel;
				secondaryOverlap = otherAdvertiser.secondChannel == secondChannel && otherAdvertiser.startTime + options.getSecondB() >= base.getTime();
			} else {
				primaryOverlap = true;
				secondaryOverlap = false;
			}

			if (otherAdvertiser.primaryChannel == primaryChannel) {
				double totalBeaconLength = options.getB() + options.getAUXOffset() + options.getSecondB();
				if (otherAdvertiser.startTime + totalBeaconLength >= base.getTime()) {
					if (myAdvertisingEventRecord.latestOverlap == 0 || 
						(myAdvertisingEventRecord.latestOverlap != 0 && myAdvertisingEventRecord.latestOverlap < otherAdvertiser.startTime)) {
						myAdvertisingEventRecord.latestOverlap = otherAdvertiser.startTime;
					}
				}
			}
	
			// handle overlap and collisions
			if (primaryOverlap || secondaryOverlap) {
				// all I have to do is mark the advertiser as a collider. Then when we end advertising, we can handle discovery.
				otherAdvertiser.collided = true;
				myAdvertisingEventRecord.collided = true;
	
				// for bookkeeping, we keep track of the collisions. So log it.
				for (Integer discovererNode : otherAdvertiser.discovererNodes) {
					int discovererNodeID = discovererNode.intValue();
					ListenEventRecord listenerRecord = getListenEventRecordForNodeID(discovererNodeID);
					if (listenerRecord != null) {
						if (logStyle == BLEDiscSimulatorOptions.LOG_STYLE_VERBOSE) {
							logger.log("C: " + base.getTime() + " : " + base.getNodeID() + " : "
									+ otherAdvertiser.nodeID + " : (" + discovererNodeID + ")\n");
						}
					}
					// we want to keep track of the collision, just for completeness
					Collision c = new Collision(base.getNodeID(), otherAdvertiser.nodeID, discovererNodeID, base.getTime());
					collisions.add(c);
				}
			}
		}
		currentExtendedAdvertisers.add(myAdvertisingEventRecord);
	}
	
	public void process(BLEExtendedAdvertiseEndEvent baee) {
		// if I didn't collide, then the current listeners (who were also listening when I started) discovered me
		ExtendedAdvertisingEventRecord aer = getExtendedAdvertisingEventRecordForNodeID(baee.getNodeID());

		boolean isExtended = options.getProtocol() == BLEDiscSimulatorOptions.METHOD_MIX || options.getProtocol() == BLEDiscSimulatorOptions.METHOD_EXTENDED || options.getProtocol() == BLEDiscSimulatorOptions.METHOD_EXTENDED_NIHAO;

		if (aer != null) {
			if (!aer.collided) {
				for (ListenEventRecord listenerRecord : currentListeners) {
					boolean isBlocked = listenerRecord.startTime < aer.latestOverlap; // the beacon can't be received because the scanner is on secondary channel when this beacon starts
					double totalBeaconLength = options.getB() + options.getAUXOffset() + options.getSecondB();
					boolean isMissed = listenerRecord.startTime >= baee.time - totalBeaconLength; // if the beacon start time is less than the listener's start time, the beacon is not received
					if (listenerRecord.startTime < aer.startTime && !isBlocked && !isMissed || !isExtended) { 
						// if we're modeling channels, we have to make sure the channels match
						if (!options.modelChannels() || aer.primaryChannel == listenerRecord.channel) {
							aer.addDiscovererNode(listenerRecord.nodeID);
							listenerRecord.addDiscoveryEvent(baee.getNodeID(), aer.startTime, baee.getSeq());
	
							// Trigger discovery event
							BLESchedule discoverersSchedule = getScheduleForNodeID(listenerRecord.nodeID);
							discoverersSchedule.onDiscovery(baee, this);
						}
					}
				}
			}
			// remove the aer
			currentExtendedAdvertisers.remove(aer);
		}
	}


    // this will grab the DiscoveryEvent object associated with a given node id
    // it's a helper method to assist the collision detection algorithm
    private ListenEventRecord getListenEventRecordForNodeID(int nodeID){
		for(ListenEventRecord listenRecord : currentListeners){
			if(listenRecord.nodeID == nodeID){
			return listenRecord;
			}
		}
		return null;
    }

    // this is similarly a helper method for me to find the right advertising event record so I can remove it
	private ExtendedAdvertisingEventRecord getExtendedAdvertisingEventRecordForNodeID(int nodeID){
		for(ExtendedAdvertisingEventRecord record : currentExtendedAdvertisers){
			if(record.nodeID == nodeID){
			return record;
			}
		}
		return null;
    }

    public BLESchedule getScheduleForNodeID(int nodeID){
		for(BLESchedule schedule : nodeSchedules){
			if(schedule.getNodeID() == nodeID){
			return schedule;
			}
		}
		return null;
    }

    @SuppressWarnings("unchecked")
    private void loadSchedules(String scheduleLoadFile){
		System.out.println("Loading schedules");
		try{
			InputStream file = new FileInputStream(scheduleLoadFile);
			InputStream buffer = new BufferedInputStream(file);
			ObjectInput ois = new ObjectInputStream(buffer);
			try{
			nodeSchedules = (ArrayList<BLESchedule>)ois.readObject();
			} catch(ClassNotFoundException cnfe){
			cnfe.printStackTrace();
			} finally{
			ois.close();
			}
		} catch(IOException ioe){
			ioe.printStackTrace();
		}
		
		makeEventQueue();
    }

    private void writeSchedules(String scheduleSaveFile){
		try{
			OutputStream file = new FileOutputStream(scheduleSaveFile);
			OutputStream buffer = new BufferedOutputStream(file);
			ObjectOutput oos = new ObjectOutputStream(buffer);
			try{
			oos.writeObject(nodeSchedules);
			} finally {
			oos.flush();
			oos.close();
			}
		} catch(IOException ioe){
			ioe.printStackTrace();
		}
    }

    private double computeDiscoveryRate(int nodeID, double fromTime){
		ArrayList<Integer> discoveredNodeIDs = new ArrayList<Integer>();
		for(CompletedDiscovery discoveryEvent : successfulDiscoveries){
			if(discoveryEvent.discovererID == nodeID){
				Integer discoveredIDInteger = Integer.valueOf(discoveryEvent.discoveredID);
				if(!discoveredNodeIDs.contains(discoveredIDInteger)){
					if(discoveryEvent.timestamp > fromTime){
					discoveredNodeIDs.add(discoveredIDInteger);
					}
				}
			}
		}
		return discoveredNodeIDs.size() / (double) (nodeSchedules.size() - 1);
    }

    private double computeAverageDiscoveryLatency(int nodeID, double fromTime){
		HashMap<Integer, Double> nodesDiscoveryEvents = new HashMap<Integer, Double>();
		for(CompletedDiscovery discoveryEvent : successfulDiscoveries){
			if(discoveryEvent.discovererID == nodeID){
			Integer discoveredIDInteger = Integer.valueOf(discoveryEvent.discoveredID);
				if(!nodesDiscoveryEvents.containsKey(discoveredIDInteger)){
					if(discoveryEvent.timestamp > fromTime){
					nodesDiscoveryEvents.put(discoveredIDInteger, Double.valueOf(discoveryEvent.timestamp - fromTime));
					}
				}
			}
		}
		
		double sumOfLatencies = nodesDiscoveryEvents.values().stream().mapToDouble(d -> d).sum();
		return sumOfLatencies / (double) nodesDiscoveryEvents.size();
    }


    

    public static void main(String[] args){
		//System.out.println("BLEDiscSimulator Starting...");
		if(args.length != 4){
			System.out.println("Usage: java BLEDiscSimulator <propertiesfile> <logfile> <numberofruns>");
		}
		else{
			int numRuns = Integer.parseInt(args[3]);
			for(int i = 0; i<numRuns; i++){
			BLEDiscSimulator simulator = new BLEDiscSimulator(args[0], args[1], args[2]);
			simulator.simulate();
			}
		}
    }


    class CompletedDiscovery{
		int discovererID;
		int discoveredID;
		double timestamp;
		int seq = -1;

		CompletedDiscovery(int discovererID, int discoveredID, double timestamp){
			this.discovererID = discovererID;
			this.discoveredID = discoveredID;
			this.timestamp = timestamp;
		}

		CompletedDiscovery(int discovererID, int discoveredID, double timestamp, int seq){
			this.discovererID = discovererID;
			this.discoveredID = discoveredID;
			this.timestamp = timestamp;
			this.seq = seq;
		}

		public int compareTo(CompletedDiscovery d2){
			return Double.compare(timestamp, d2.timestamp);   
		}
    }

    class Collision{
		int node1;
		int node2;
		int listener;
		double timestamp;

		Collision(int node1, int node2, int listener, double timestamp){
			this.node1 = node1;
			this.node2 = node2;
			this.timestamp = timestamp;
		}
    }

    // this class is used by the listening task to keep track of nodes that are discovered
    // (and timestamps for that discovery)
    class ListenEventRecord{

		int nodeID;
		int channel;
		double startTime;
		ArrayList<DiscoveryEvent> discoveryEvents = new ArrayList<DiscoveryEvent>();

		ListenEventRecord(int nodeID, int channel, double startTime){
			this.nodeID = nodeID;
			this.channel = channel;
			this.startTime = startTime;
		}

		void addDiscoveryEvent(int nodeID, double timestamp){
			discoveryEvents.add(new DiscoveryEvent(nodeID, timestamp));
		}

		void addDiscoveryEvent(int nodeID, double timestamp, int seq){
			discoveryEvents.add(new DiscoveryEvent(nodeID, timestamp, seq));
		}

		void removeDiscoveryEvent(int nodeID){
			for(int i = 0; i<discoveryEvents.size(); i++){
			if(discoveryEvents.get(i).discoveredNode == nodeID){
				discoveryEvents.remove(i);
			}
			}
		}

		class DiscoveryEvent{
			int discoveredNode;
			double timestamp;
			int seq = -1; // sequence of discovered chunk

			DiscoveryEvent(int discoveredNode, double timestamp){
			this.discoveredNode = discoveredNode;
			this.timestamp = timestamp;
			}

			DiscoveryEvent(int discoveredNode, double timestamp, int seq){
				this.discoveredNode = discoveredNode;
				this.timestamp = timestamp;
				this.seq = seq;
			}
		}
	    
	
    }

    // this class is used in the advertising task to keep track of the nodes that have discovered
    // the advertiser. Note that this is not really something that an advertiser can KNOW, but we
    // use it simply to keep track of collisions and to undo discovery events after the fact
	// HC: CHANGING!
	class ExtendedAdvertisingEventRecord{

		int nodeID;
		boolean collided;
		int primaryChannel;
		int secondChannel;
		double startTime;
		double latestOverlap; // The latest start time among all advertising events that begin prior to and overlap with the current event.
		ArrayList<Integer> discovererNodes = new ArrayList<Integer>();

		ExtendedAdvertisingEventRecord(int nodeID, int primaryChannel, int secondChannel, double startTime){
			this.nodeID = nodeID;
			this.primaryChannel = primaryChannel;
			this.secondChannel = secondChannel;
			this.startTime = startTime;
		}

		void addDiscovererNode(int nodeID){
			discovererNodes.add(Integer.valueOf(nodeID));
		}

		void removeDiscovererNode(int nodeID){
			discovererNodes.remove(Integer.valueOf(nodeID));
		}
    }
}
