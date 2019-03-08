
package ijp.proxy;

import ijp.Picture;
import ijp.service.Service;
import ijp.service.ServiceFromProperties;

import java.util.HashMap;
import java.util.Arrays;
import java.util.List;

/**
* A proxy service for the PictureViewer application which caches photos
* in case of a slow proxy. Code modified from RetryProxy.
*
* @author  Chris Sipola
* @version 1.0
*/

public class MyCacheProxy implements Service {
	
	private Service base_service = null;
	
	// instantiate picture cache as List because Array doesn't behave as expected as the key
	private HashMap<List<String>, Picture> pictureCache = new HashMap<List<String>, Picture>();
	
	/**
	 * Return a textual name for the service. (Note to grader: I'm putting
	 * this at the top to keep it consistent with RetryProxy. Otherwise I'd
	 * put it at the bottom or at least after the constructors.)
	 *
	 * @return the name of the service
	 */
	public String serviceName() {
		return base_service.serviceName();
	}
	
	/**
	 * Constructor with input base service.
	 * 
	 * @param base_service the base service 
	 */
	public MyCacheProxy(Service base_service) {
		this.base_service = base_service;
	}
	
	/**
	 * Constructor that uses base service from my.properties file.
	 */
	public MyCacheProxy() {
		base_service = new ServiceFromProperties("MyCacheProxy.base_service");
	}
	
	/**
	 * Get picture from service and store in cache (if it hasn't yet been selected)
	 * or from the cache (if it's already been selected). Note that a picture is
	 * identified by both a subject and an index.
	 * 
	 * @param subject the name of the subject
	 * @param index the index of the subject
	 * 
	 * @return picture the picture for given subject and index
	 */
	public Picture getPicture(String subject, int index) {
		List<String> subjectAndIndex = Arrays.asList(subject, String.valueOf(index));
		Picture picture = pictureCache.get(subjectAndIndex);
		if (picture == null) {
			picture = base_service.getPicture(subject, index);
			pictureCache.put(subjectAndIndex, picture);
		}
		return picture;
	}
}
