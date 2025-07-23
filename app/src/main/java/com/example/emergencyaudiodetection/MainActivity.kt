package com.example.emergencyaudiodetection

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.*
import android.telephony.SmsManager
import android.telephony.SubscriptionManager
import android.util.Log
import android.widget.Button
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import be.tarsos.dsp.AudioDispatcher
import be.tarsos.dsp.AudioEvent
import be.tarsos.dsp.AudioProcessor
import be.tarsos.dsp.io.android.AudioDispatcherFactory
import com.example.emergencyaudiodetection.contact.ManageContactsActivity
import com.google.android.gms.location.LocationServices
import kotlinx.coroutines.*
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.IOException
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class MainActivity : AppCompatActivity() {

    private lateinit var recordToggleButton: Button
    private lateinit var manageContactsButton: Button

    private var dispatcher: AudioDispatcher? = null
    private var isRecording = false
    private var tflite: Interpreter? = null
    private var yamnet: Interpreter? = null

    private var alertDialog: android.app.AlertDialog? = null
    private var autoSendAlertHandler: Handler? = null
    private var autoSendAlertRunnable: Runnable? = null
    private val confidenceWindow = ArrayDeque<Float>() // ✅ ADDED: For smoothing
    private val smoothingWindowSize = 7


    private val coroutineScope = CoroutineScope(Dispatchers.IO)

    companion object {
        private const val SAMPLE_RATE = 16000
        private const val INPUT_LENGTH = 15600
        private const val YAMNET_EMBEDDING_SIZE = 1024
        private const val CONFIDENCE_THRESHOLD = 0.8f
        private const val PERMISSION_REQUEST_CODE = 100
        private const val AUTO_SEND_DELAY_MS = 10000L // 10 seconds
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        recordToggleButton = findViewById(R.id.recordButton)
        manageContactsButton = findViewById(R.id.manageContactsButton)

        recordToggleButton.setOnClickListener {
            if (!isRecording) {
                startRecording()
            } else {
                stopRecording()
            }
        }

        manageContactsButton.setOnClickListener {
            startActivity(Intent(this, ManageContactsActivity::class.java))
        }

        if (!hasAllPermissions()) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(
                    Manifest.permission.RECORD_AUDIO,
                    Manifest.permission.SEND_SMS,
                    Manifest.permission.ACCESS_FINE_LOCATION
                ),
                PERMISSION_REQUEST_CODE
            )
        }

        try {
            yamnet = Interpreter(loadModelFile("yamnet_embedding.tflite"))
            tflite = Interpreter(loadModelFile("emergency_audio_model.tflite"))
            // Log the input tensor shape
            val inputShape = yamnet!!.getInputTensor(0).shape()
            // E.g., [15600] or [1, 15600]
            Log.d("YAMNetOutputShape", yamnet!!.getOutputTensor(0).shape().joinToString())
            Log.d("ClassifierInputShape", tflite!!.getInputTensor(0).shape().joinToString())
            Log.d("ClassifierOutputShape", tflite!!.getOutputTensor(0).shape().joinToString())


        } catch (e: IOException) {
            Log.e("ModelInit", "Failed to load model", e)
            Toast.makeText(this, "Model loading error!", Toast.LENGTH_LONG).show()
        }
    }

    private fun hasAllPermissions(): Boolean {
        return ActivityCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED &&
                ActivityCompat.checkSelfPermission(this, Manifest.permission.SEND_SMS) == PackageManager.PERMISSION_GRANTED &&
                ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED
    }

    private fun loadModelFile(modelName: String): MappedByteBuffer {
        assets.openFd(modelName).apply {
            return FileInputStream(fileDescriptor).channel.map(
                FileChannel.MapMode.READ_ONLY,
                startOffset,
                declaredLength
            )
        }
    }

    private fun startRecording() {
        if (!hasAllPermissions()) {
            Toast.makeText(this, "Missing permissions.", Toast.LENGTH_LONG).show()
            return
        }
        if (yamnet == null || tflite == null) {
            Toast.makeText(this, "Models not loaded!", Toast.LENGTH_LONG).show()
            return
        }

        Log.d("Recording", "Start recording")
        dispatcher = AudioDispatcherFactory.fromDefaultMicrophone(SAMPLE_RATE, 15360, 7680)

        dispatcher?.addAudioProcessor(object : AudioProcessor {
            override fun processingFinished() {}

            override fun process(audioEvent: AudioEvent): Boolean = try {
                val audioBuffer = audioEvent.floatBuffer
                val yamnetInput = FloatArray(INPUT_LENGTH)
                for (i in audioBuffer.indices) {
                    if (i < INPUT_LENGTH) {
                        yamnetInput[i] = audioBuffer[i]
                    }
                }


                // Get the embedding
                val embeddingOutput = FloatArray(1024)
                yamnet?.run(yamnetInput, embeddingOutput)

                // Wrap embeddingOutput for classifier (batch of 1)
                val classifierInput = arrayOf(embeddingOutput)
                val classifierOutput = Array(1) { FloatArray(1) }
                tflite?.run(classifierInput, classifierOutput)
                val confidence = classifierOutput[0][0]
                val cappedConfidence = confidence.coerceAtMost(0.85f)
                Log.d("InferenceOutput", "Confidence: $confidence")

                // ✅ ADD CONFIDENCE SMOOTHING
                confidenceWindow.addLast(cappedConfidence)
                if (confidenceWindow.size > smoothingWindowSize) {
                    confidenceWindow.removeFirst()
                }
                val avgConfidence = confidenceWindow.average().toFloat()
                Log.d("InferenceOutput", "Smoothed Confidence: $avgConfidence")
                Log.d("InferenceOutput", "Confidence: $confidence")
                Log.d("InferenceOutput", "Capped Confidence: $cappedConfidence")
                Log.d("InferenceOutput", "Smoothed Confidence: $CONFIDENCE_THRESHOLD")


                if (avgConfidence > CONFIDENCE_THRESHOLD) {
                    runOnUiThread {
                        Toast.makeText(applicationContext, "🚨 Emergency Sound Detected!", Toast.LENGTH_SHORT).show()
                        if (alertDialog?.isShowing != true) {
                            triggerAlert()
                        }
                    }
                }

                true
            } catch (e: Exception) {
                Log.e("InferenceError", "YAMNet failed:", e)
                true
            }
        })

        isRecording = true
        coroutineScope.launch {
            dispatcher?.run()
        }

        recordToggleButton.text = getString(R.string.stop_listening)
        recordToggleButton.setBackgroundResource(R.drawable.btn_shape_disabled)
        Toast.makeText(this, "Recording started", Toast.LENGTH_SHORT).show()
    }

    private fun stopRecording() {
        isRecording = false
        dispatcher?.stop()
        dispatcher = null
        recordToggleButton.text = getString(R.string.start_listening)
        recordToggleButton.setBackgroundResource(R.drawable.btn_shape_enabled)
        Toast.makeText(this, "Recording stopped", Toast.LENGTH_SHORT).show()
        dismissAlertDialog()
    }

    private fun triggerAlert() {
        alertDialog = android.app.AlertDialog.Builder(this)
            .setTitle("🚨 Emergency Detected")
            .setMessage("Emergency detected! Send alert to your contacts?")
            .setCancelable(false)
            .setPositiveButton("Yes") { _, _ -> sendAlert() }
            .setNegativeButton("No") { _, _ ->
                Toast.makeText(this, "Alert cancelled.", Toast.LENGTH_SHORT).show()
            }
            .create()

        alertDialog?.show()

        // Auto-send if no response
        autoSendAlertHandler = Handler(Looper.getMainLooper())
        autoSendAlertRunnable = Runnable {
            if (alertDialog?.isShowing == true) {
                alertDialog?.dismiss()
                sendAlert()
            }
        }
        autoSendAlertHandler?.postDelayed(autoSendAlertRunnable!!, AUTO_SEND_DELAY_MS)
    }

    private fun dismissAlertDialog() {
        autoSendAlertHandler?.removeCallbacks(autoSendAlertRunnable ?: return)
        alertDialog?.dismiss()
    }


    private fun sendAlert() {
        Log.d("SMS", "sendAlert() called")

        // Debug contacts
        debugContacts()

        // Test SMS permission
        if (!testSMSPermission()) {
            Toast.makeText(this, "SMS permission required", Toast.LENGTH_SHORT).show()
            return
        }

        val sharedPreferences = getSharedPreferences("EmergencyContacts", MODE_PRIVATE)
        val contactSet = HashSet(sharedPreferences.getStringSet("contacts", emptySet()) ?: emptySet())

        Log.d("ContactsDebug", "Loaded contacts: $contactSet")

        if (contactSet.isEmpty()) {
            Toast.makeText(this, "No emergency contacts saved. Please add contacts first.", Toast.LENGTH_LONG).show()
            return
        }

        // Get location and send SMS
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED) {
            val fusedLocationClient = LocationServices.getFusedLocationProviderClient(this)
            fusedLocationClient.lastLocation.addOnSuccessListener { location ->
                val locationText = location?.let {
                    "https://maps.google.com/?q=${it.latitude},${it.longitude}"
                } ?: "Location unavailable"

                Log.d("Location", "Location: $locationText")
                sendSmsToContacts(contactSet, locationText)
            }.addOnFailureListener { exception ->
                Log.e("Location", "Failed to get location", exception)
                sendSmsToContacts(contactSet, "Location unavailable")
            }
        } else {
            Log.w("Location", "Location permission not granted")
            sendSmsToContacts(contactSet, "Location unavailable")
        }
    }

    private fun sendSmsToContacts(contacts: Set<String>, location: String) {
        Log.d("SMS", "Attempting to send SMS to ${contacts.size} contacts")

        if (contacts.isEmpty()) {
            Toast.makeText(this, "No emergency contacts found", Toast.LENGTH_SHORT).show()
            return
        }

        // Check SMS permission
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.SEND_SMS) != PackageManager.PERMISSION_GRANTED) {
            Toast.makeText(this, "SMS permission not granted", Toast.LENGTH_SHORT).show()
            return
        }

        val smsManager: SmsManager = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            getSystemService(SmsManager::class.java)
        } else {
            SmsManager.getDefault()
        }

        val message = "🚨 EMERGENCY ALERT!\nI may be in danger and need help.\nLocation: $location\nPlease contact me or authorities immediately."

        var successCount = 0
        var failureCount = 0

        contacts.forEach { contact ->
            try {
                Log.d("SMS", "Sending SMS to: $contact")

                // Split long messages if needed
                val parts = smsManager.divideMessage(message)
                if (parts.size == 1) {
                    smsManager.sendTextMessage(contact, null, message, null, null)
                } else {
                    smsManager.sendMultipartTextMessage(contact, null, parts, null, null)
                }

                successCount++
                Log.d("SMS", "SMS sent successfully to: $contact")

            } catch (e: Exception) {
                failureCount++
                Log.e("SMS", "Failed to send SMS to $contact: ${e.message}", e)
            }
        }

        runOnUiThread {
            if (successCount > 0) {
                Toast.makeText(this, "🚨 Emergency alerts sent to $successCount contact(s)!", Toast.LENGTH_LONG).show()
            }
            if (failureCount > 0) {
                Toast.makeText(this, "Failed to send to $failureCount contact(s)", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun debugContacts() {
        val sharedPreferences = getSharedPreferences("EmergencyContacts", MODE_PRIVATE)
        val contactSet = HashSet(sharedPreferences.getStringSet("contacts", emptySet()) ?: emptySet())

        Log.d("ContactsDebug", "Total contacts: ${contactSet.size}")
        contactSet.forEachIndexed { index, contact ->
            Log.d("ContactsDebug", "Contact $index: '$contact'")
        }
    }

    private fun testSMSPermission(): Boolean {
        val hasPermission = ActivityCompat.checkSelfPermission(this, Manifest.permission.SEND_SMS) == PackageManager.PERMISSION_GRANTED
        Log.d("SMS", "SMS Permission granted: $hasPermission")
        return hasPermission
    }

    
    override fun onDestroy() {
        super.onDestroy()
        coroutineScope.cancel()
        yamnet?.close()
        tflite?.close()
        dispatcher?.stop()
        autoSendAlertHandler?.removeCallbacks(autoSendAlertRunnable ?: return)
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, results: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, results)
        if (requestCode == PERMISSION_REQUEST_CODE) {
            if (!hasAllPermissions()) {
                Toast.makeText(this, "All permissions are required for emergency detection.", Toast.LENGTH_LONG).show()
            }
        }
    }
}
