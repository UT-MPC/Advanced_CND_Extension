import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.BufferedWriter;

public class BLEDiscLogger{

    private BufferedWriter bw;
    
    public BLEDiscLogger(String logfile){
	try{
	    File f = new File(logfile);
	    if(!f.exists()){
		f.createNewFile();
	    }
	    FileWriter writer = new FileWriter(f.getAbsoluteFile(), true);
	    bw = new BufferedWriter(writer);
	}catch(IOException ioe){
	    ioe.printStackTrace();
	}
    }

    public void log(String logMessage){
	try{
	    bw.write(logMessage);
	} catch(IOException ioe){
	    ioe.printStackTrace();
	}
    }

    public void close(){
	try{
	    bw.close();
	} catch(IOException ioe){
	    ioe.printStackTrace();
	}
    }
}
