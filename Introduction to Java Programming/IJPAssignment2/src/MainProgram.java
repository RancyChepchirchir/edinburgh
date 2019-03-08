import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;

import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.scene.layout.AnchorPane;
import javafx.stage.Stage;

import javafx.application.Application;

/**
 * Class MainProgram - runs the main program.
 * 
 * Loads the FXML file, sets the scene, and creates the Controller object.
 * 
 * The code was taken directly from the IJP course website with just a few
 * modifications.
 * 
 * @author Chris Sipola (s1667278)
 * @version 2016.11.25
 */
public class MainProgram extends Application {

	public void start(Stage stage) {

		try {

			FXMLLoader fxmlLoader = new FXMLLoader();
			String viewerFxml = "WorldViewer.fxml";
			AnchorPane page = (AnchorPane) fxmlLoader.load(this.getClass().getResource(viewerFxml).openStream());
			Scene scene = new Scene(page);
			stage.setScene(scene);

			Controller controller = (Controller) fxmlLoader.getController();
			controller.Initialize();

			stage.show();

		} catch (IOException ex) {
			Logger.getLogger(this.getClass().getName()).log(Level.SEVERE, null, ex);
			System.exit(1);
		}
	}

	public static void main(String args[]) {
		launch(args);
		System.exit(0);
	}
}
